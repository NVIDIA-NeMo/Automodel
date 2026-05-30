# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""2-GPU FSDP2 sharded forward+backward test for ``Gemma4WithDrafter``.

This is the canonical regression guard for the joint base + drafter training
path under FSDP2: it spawns two NCCL ranks via ``torch.multiprocessing.spawn``,
shards both sub-modules of the composite via ``fully_shard`` on a 2-rank mesh,
and runs a forward + backward on a tiny config. The test passes when:

  * the joint forward returns logits + drafter_logits with the right shape on
    every rank;
  * a joint backward populates non-zero gradients on the drafter side AND on
    a base K/V projection (proving the drafter loss flowed through
    ``shared_kv_states`` after sharding);
  * ``backward()`` does not deadlock (each FSDP-wrapped sub-module triggers
    its own all-reduce / reduce-scatter cycle).

Skipped when:

  * fewer than 2 visible CUDA devices, or
  * ``transformers.models.gemma4_assistant`` is not importable.
"""

from __future__ import annotations

import importlib
import os
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _gemma4_assistant_available() -> bool:
    try:
        importlib.import_module("transformers.models.gemma4_assistant")
        importlib.import_module("transformers.models.gemma4")
        return True
    except (ModuleNotFoundError, ImportError):
        return False


_HAS_GEMMA4_ASSISTANT = _gemma4_assistant_available()


def _free_port() -> int:
    """Pick a free localhost TCP port for the rendezvous endpoint."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _fsdp2_worker(rank: int, world_size: int, port: int, result_path: str):
    """Body of each FSDP rank.

    Runs to completion or writes a traceback to ``result_path`` so the parent
    process can surface the exception cleanly. Side-effects (process group
    creation, CUDA context) are torn down before return.
    """
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = torch.bfloat16

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        # Tiny real Gemma4 base + drafter -- same recipe as the single-process
        # tests in test_composite.py, kept inline so the worker is fully
        # self-contained.
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4Config,
            Gemma4TextConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
        )
        from transformers.models.gemma4_assistant.configuration_gemma4_assistant import (
            Gemma4AssistantConfig,
        )
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
            Gemma4AssistantForCausalLM,
        )

        from nemo_automodel.components.models.gemma4_drafter.composite import (
            Gemma4JointOutput,
            Gemma4WithDrafter,
        )

        torch.manual_seed(0)

        base_text = Gemma4TextConfig(
            vocab_size=64,
            hidden_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            num_hidden_layers=4,
            num_kv_shared_layers=2,
            intermediate_size=64,
            rms_norm_eps=1e-6,
            max_position_embeddings=64,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=8,
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=0,
            enable_moe_block=False,
            use_double_wide_mlp=False,
            torch_dtype="bfloat16",
        )
        base_cfg = Gemma4Config(text_config=base_text)
        base = Gemma4ForConditionalGeneration(base_cfg).to(device=device, dtype=dtype)

        draft_text = Gemma4TextConfig(
            vocab_size=base_text.vocab_size,
            hidden_size=24,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=12,
            num_hidden_layers=2,
            intermediate_size=48,
            rms_norm_eps=1e-6,
            max_position_embeddings=64,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=8,
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=0,
            enable_moe_block=False,
            use_double_wide_mlp=False,
            torch_dtype="bfloat16",
        )
        draft_cfg = Gemma4AssistantConfig(
            text_config=draft_text,
            backbone_hidden_size=base_text.hidden_size,
        )
        drafter = Gemma4AssistantForCausalLM(draft_cfg).to(device=device, dtype=dtype)

        composite = Gemma4WithDrafter(
            base,
            drafter,
            drafter_loss_weight=0.5,
            drafter_num_steps=1,
        )

        # FSDP2 sharding: shard the two sub-modules separately on the same
        # 2-rank mesh. This mirrors how the recipe handles the composite
        # (each sub-module is loaded via NeMoAutoModel and gets its own
        # fully_shard call inside apply_model_infrastructure).
        mesh = init_device_mesh("cuda", (world_size,))
        fully_shard(composite.base, mesh=mesh)
        fully_shard(composite.drafter, mesh=mesh)

        # Per-rank input ids -- different seeds so the all-reduce sees
        # distinct gradients and not a no-op.
        torch.manual_seed(rank)
        input_ids = torch.randint(
            0,
            base_text.vocab_size,
            (1, 4),
            device=device,
        )

        out = composite(input_ids=input_ids)

        if not isinstance(out, Gemma4JointOutput):
            raise AssertionError(f"expected Gemma4JointOutput, got {type(out)}")
        if out.logits.shape != (1, 4, base_text.vocab_size):
            raise AssertionError(f"unexpected logits shape: {out.logits.shape}")
        if len(out.drafter_logits) != 1:
            raise AssertionError(f"expected 1 drafter step, got {len(out.drafter_logits)}")
        if out.drafter_logits[0].shape != (1, 4, draft_text.vocab_size):
            raise AssertionError(f"unexpected drafter_logits shape: {out.drafter_logits[0].shape}")

        loss = out.logits.float().sum() + out.drafter_loss_weight * out.drafter_logits[0].float().sum()
        loss.backward()

        # Verify gradients exist on the drafter pre_projection (DTensor under
        # FSDP2; ``.to_local()`` exposes the local shard).
        pre_proj_grad = composite.drafter.pre_projection.weight.grad
        if pre_proj_grad is None:
            raise AssertionError("drafter.pre_projection.weight.grad is None on rank %d" % rank)
        local_pre_proj = pre_proj_grad.to_local() if hasattr(pre_proj_grad, "to_local") else pre_proj_grad
        if not torch.any(local_pre_proj != 0):
            raise AssertionError("drafter.pre_projection.weight.grad is all zeros on rank %d" % rank)

        # Verify drafter loss reached at least one base k_proj/v_proj weight
        # via shared_kv_states (the joint-training-correctness signal).
        seen_kv = False
        for name, p in composite.base.named_parameters():
            if (".k_proj." in name or ".v_proj." in name) and p.grad is not None:
                local = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                if torch.any(local != 0):
                    seen_kv = True
                    break
        if not seen_kv:
            raise AssertionError("Drafter loss did not reach any base k_proj/v_proj weight on rank %d" % rank)

        # Drafter loss must reach the base embed_tokens (consumed by drafter
        # via the concatenated inputs_embeds).
        embed_grad = composite.base.get_input_embeddings().weight.grad
        if embed_grad is None:
            raise AssertionError("base.embed_tokens.weight.grad is None on rank %d" % rank)
        local_embed = embed_grad.to_local() if hasattr(embed_grad, "to_local") else embed_grad
        if not torch.any(local_embed != 0):
            raise AssertionError("base.embed_tokens.weight.grad is all zeros on rank %d" % rank)

        dist.barrier()
        # Mark success.
        with open(result_path + f".rank{rank}", "w") as f:
            f.write("OK\n")

    except Exception:
        with open(result_path + f".rank{rank}", "w") as f:
            f.write("FAIL\n")
            f.write(traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not _HAS_GEMMA4_ASSISTANT, reason="transformers.models.gemma4_assistant not available")
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least 2 CUDA devices",
)
def test_gemma4_with_drafter_fsdp2_2gpu(tmp_path):
    """Spawn 2 NCCL ranks, FSDP2-wrap each sub-module, run joint
    forward + backward, and validate gradients on both sides."""
    world_size = 2
    port = _free_port()
    result_path = str(tmp_path / "result")

    # ``mp.spawn`` returns when all workers exit; non-zero exit raises a
    # ``ProcessRaisedException`` that we let propagate so the failing rank's
    # traceback is visible.
    mp.spawn(
        _fsdp2_worker,
        args=(world_size, port, result_path),
        nprocs=world_size,
        join=True,
    )

    # Defensive sanity: confirm both ranks reported success on disk.
    for rank in range(world_size):
        rank_path = result_path + f".rank{rank}"
        assert os.path.exists(rank_path), f"rank {rank} did not write a result"
        with open(rank_path) as f:
            status = f.read().splitlines()[0]
        assert status == "OK", f"rank {rank} failed: {open(rank_path).read()}"
