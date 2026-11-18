# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Numerical isolation against the real released remote-code models on one GPU.

The configurations and remote code come from the publisher revisions that the
packing adapters target; no weights are downloaded. Set
LEGACY_PACKING_FLASH_CHECKPOINT / LEGACY_PACKING_NAS_CHECKPOINT to use local
snapshots instead. Only model dimensions shrink; the actual released layers,
attention, recurrent kernels, memory prefixes and checkpoint names are used.
"""

import importlib
import importlib.util
import math
import os
import sys
from collections.abc import Iterator
from datetime import timedelta
from typing import Protocol, cast

import pytest
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from triton import Config
from triton.runtime.autotuner import Autotuner

from nemo_automodel import NeMoAutoConfig, NeMoAutoModelForCausalLM
from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.datasets.utils import neat_packed_collater
from nemo_automodel.components.distributed.pipelining.hf_utils import patch_hf_model_for_pp
from nemo_automodel.components.models.common.packing import configure_packing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# Publisher revisions whose remote-code interfaces the packing adapters target.
_RELEASES = {
    "flash": ("nvidia/Nemotron-Flash-1B", "da87f707f678b5936e67fdcfe598f76ebcc5fdbc"),  # pragma: allowlist secret
    "nas": ("nvidia/Llama-3_3-Nemotron-Super-49B-v1_5", "420ba7d28211abf116b8b103ab700d92619daf98"),  # pragma: allowlist secret
}


def _footprint(config: Config) -> tuple[int, int, int]:
    blocks = math.prod(value for value in config.kwargs.values() if isinstance(value, int))
    return config.num_stages, blocks, config.num_warps


@pytest.fixture(scope="module", autouse=True)
def single_triton_configs() -> Iterator[None]:
    # Cold autotuning compiles every candidate configuration of every FLA and
    # Mamba kernel, which dominated this module's CI time. Parity does not depend
    # on the launch configuration, so compile only the smallest-footprint
    # candidate; Triton skips benchmarking when one configuration remains.
    for package in ("fla.modules", "fla.ops.delta_rule", "mamba_ssm.ops.triton.ssd_combined"):
        if importlib.util.find_spec(package.partition(".")[0]) is not None:
            importlib.import_module(package)
    original: dict[Autotuner, list[Config]] = {}
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith(("fla.", "mamba_ssm.")):
            continue
        for value in vars(module).values():
            for _ in range(3):  # Heuristics and Autotuner wrappers nest via .fn.
                if isinstance(value, Autotuner) and len(value.configs) > 1 and value not in original:
                    original[value] = value.configs
                    value.configs = [min(value.configs, key=_footprint)]
                value = getattr(value, "fn", None)
    yield
    for tuner, configs in original.items():
        tuner.configs = configs


@pytest.fixture(scope="module", autouse=True)
def distributed_context(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    apply_cache_compatibility_patches()
    created = not dist.is_initialized()
    if created:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        # CI collects this module with plain pytest, so rendezvous through a private file.
        rendezvous = tmp_path_factory.mktemp("nccl") / "rendezvous"
        dist.init_process_group(
            "nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1, timeout=timedelta(seconds=120)
        )
    yield
    if created:
        dist.destroy_process_group()


def _release_path(family: str) -> str:
    path = os.environ.get(f"LEGACY_PACKING_{family.upper()}_CHECKPOINT")
    if path:
        return path
    repo_id, revision = _RELEASES[family]
    try:
        return snapshot_download(repo_id, revision=revision, allow_patterns=["config.json", "*.py"])
    except LocalEntryNotFoundError as error:
        pytest.skip(f"{repo_id}@{revision} is neither cached nor reachable on the Hugging Face Hub: {error}")


def _model(family: str, attn_implementation: str = "flash_attention_2") -> torch.nn.Module:
    cfg = NeMoAutoConfig.from_pretrained(_release_path(family), trust_remote_code=True)
    data = cfg.to_dict()
    data.update(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )
    if family == "flash":
        data.update(
            num_hidden_layers=4,
            num_memory_tokens=4,
            layer_types=["deltanet", "f", "m2", "a"],
            mamba2_headdim=32,
            mamba_d_state=16,
            mamba_dt_rank=8,
            attn_implementation_new=attn_implementation,
            attn_implementation=attn_implementation,
        )
    else:
        blocks = [data["block_configs"][i] for i in (0, 1, 6, 7)]
        for block in blocks:
            if not block["attention"]["no_op"]:
                block["attention"]["n_heads_in_group"] = 2
        data.update(num_hidden_layers=4, block_configs=blocks)
    cfg = type(cfg).from_dict(data)
    torch.manual_seed(123)
    model = (
        NeMoAutoModelForCausalLM.from_config(
            cfg,
            force_hf=True,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            use_liger_kernel=False,
            use_sdpa_patching=False,
        )
        .cuda()
        .eval()
    )
    if family == "flash":
        # A randomly initialized padding row is zero, while this release
        # normalizes its tied output head by each row's norm. Populate that row
        # as a loaded checkpoint would, so the reference itself is finite.
        with torch.no_grad():
            model.get_input_embeddings().weight[0].normal_(std=cfg.initializer_range)
        # Use FLA's fused_recurrent delta rule, which the release selects for
        # short inputs. The adapter passes chunk mode the same cu_seqlens, but its
        # kernels take minutes to compile cold in CI.
        for kind, layer in zip(cfg.layer_types, model.model.layers, strict=True):
            if kind == "deltanet":
                layer.gla.mode = "fused_recurrent"
    return model


def _batch(model: torch.nn.Module) -> dict[str, torch.Tensor | int]:
    contract = configure_packing("flash_attention_2", model=model)
    rows = []
    for a, b, offset in [(80, 96, 0), (96, 80, 32)]:
        tokens = ((torch.arange(a + b) + 5 + offset) % 200 + 3).long()
        rows.append(
            {
                "input_ids": torch.cat((tokens, torch.zeros(80, dtype=torch.long))),
                "labels": torch.cat((tokens, torch.full((80,), -100))),
                "position_ids": torch.cat((torch.arange(a), torch.arange(b), torch.zeros(80))).long(),
                "attention_mask": torch.tensor([1] * a + [2] * b + [0] * 80),
            }
        )
    batch = neat_packed_collater(rows, packing=contract)
    batch.pop("labels")
    return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class _HasLogits(Protocol):
    logits: torch.Tensor


def _logits(output: torch.Tensor | _HasLogits) -> torch.Tensor:
    """Return a causal LM's logits.

    Args:
        output: Tensor of shape [batch, sequence, vocab], or a model output
            whose ``logits`` has that shape.

    Returns:
        Tensor of shape [batch, sequence, vocab]; the input itself, not a copy.
    """
    return output if isinstance(output, torch.Tensor) else output.logits


@pytest.mark.parametrize("family", ["flash", "nas"])
@pytest.mark.parametrize("with_lora", [False, True])
def test_packed_logits_gradients_and_document_isolation(family: str, with_lora: bool) -> None:
    model = _model(family)
    if with_lora:
        apply_lora_to_linear_modules(
            model,
            PeftConfig(match_all_linear=True, dim=8, alpha=8, use_triton=family == "nas"),
        )
        # Nonzero adapters exercise both LoRA projections, including Mamba's
        # output projection, which must remain a module call in the fused path.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("lora_B.weight"):
                    parameter.normal_(std=0.02)
        model.eval()
    batch = _batch(model)
    ids = cast(torch.Tensor, batch["input_ids"])
    documents = [(0, 0, 80), (0, 80, 176), (1, 0, 96), (1, 96, 176)]
    # A packed document must reproduce the same remote model run independently,
    # including Flash-1B's learned prefix and its DeltaNet/Mamba convolutions.
    model.zero_grad(set_to_none=True)
    references = [
        _logits(model(input_ids=ids[row : row + 1, start:end], use_cache=False)) for row, start, end in documents
    ]
    assert all(torch.isfinite(x).all() for x in references)
    reference_loss = sum(x.float().square().sum() for x in references) / sum(x.numel() for x in references)
    reference_loss.backward()
    reference_gradients = {
        k: p.grad.detach().float().clone() for k, p in model.named_parameters() if p.grad is not None
    }
    if with_lora and family == "flash":
        assert reference_gradients["model.layers.2.mamba.out_proj.lora_B.weight"].norm() > 0
    expected = [x.detach() for x in references]
    del references, reference_loss
    model.zero_grad(set_to_none=True)
    packed = _logits(model(**batch))
    slices = [packed[row : row + 1, start:end] for row, start, end in documents]
    for actual, reference in zip(slices, expected, strict=True):
        torch.testing.assert_close(actual.float(), reference.float(), atol=3e-3, rtol=3e-2)
    loss = sum(x.float().square().sum() for x in slices) / sum(x.numel() for x in slices)
    loss.backward()
    for name, parameter in model.named_parameters():
        if name in reference_gradients:
            gradient = parameter.grad
            assert gradient is not None
            torch.testing.assert_close(
                gradient.float(), reference_gradients[name], atol=5e-4, rtol=5e-2, msg=lambda msg: f"{name}: {msg}"
            )
            error = (gradient.float() - reference_gradients[name]).norm()
            assert error <= 0.05 * reference_gradients[name].norm() + 1e-7, name
    # Same tensor geometry and boundaries: changing document A must have no
    # effect at all on the later document or on the other packed row.
    changed_ids = ids.clone()
    changed_ids[0, :80] = (changed_ids[0, :80] + 7) % 200 + 3
    changed = {**batch, "input_ids": changed_ids}
    with torch.no_grad():
        perturbed = _logits(model(**changed))
    torch.testing.assert_close(perturbed[0, 80:176], packed.detach()[0, 80:176], atol=0, rtol=0)
    torch.testing.assert_close(perturbed[1, :176], packed.detach()[1, :176], atol=0, rtol=0)
    if family == "nas":
        # The shared PP forward must carry boundaries to the legacy attention
        # adapter too, rather than falling back to an ordinary causal mask.
        patch_hf_model_for_pp(model)
        with torch.no_grad():
            pipeline_logits = _logits(model(**batch))
        for row, start, end in documents:
            torch.testing.assert_close(
                pipeline_logits[row, start:end].float(), packed.detach()[row, start:end].float(), atol=3e-3, rtol=3e-2
            )


def test_nas_packing_keeps_release_precision_with_fp32_parameters_under_autocast() -> None:
    # FP32 embeddings build FP32 RoPE tables, which promote the BF16 query and
    # key; the release casts them back before FlashAttention, so packing must too.
    model = _model("nas").float()
    batch = _batch(model)
    ids = cast(torch.Tensor, batch["input_ids"])
    documents = [(0, 0, 80), (0, 80, 176), (1, 0, 96), (1, 96, 176)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        references = [
            _logits(model(input_ids=ids[row : row + 1, start:end], use_cache=False)) for row, start, end in documents
        ]
        packed = _logits(model(**batch))
    for (row, start, end), reference in zip(documents, references, strict=True):
        torch.testing.assert_close(packed[row : row + 1, start:end].float(), reference.float(), atol=3e-3, rtol=3e-2)


def test_flash_non_flash_backend_constructs_and_rejects_packing() -> None:
    model = _model("flash", "sdpa")
    assert type(model.model.layers[-1].self_attn).__name__ == "NemotronFlashSDPAAttention"
    with pytest.raises(ValueError, match="requires flash_attention_2"):
        configure_packing("sdpa", model=model)
