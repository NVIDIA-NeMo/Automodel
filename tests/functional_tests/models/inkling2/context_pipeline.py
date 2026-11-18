# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Four-GPU numerical qualification of image PP2/EP2/CP2 through public factories."""

import copy
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel.components.datasets.vlm.pp_media import prepare_vlm_media_for_pp, stage_vlm_media_for_pp
from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.config import Inkling2Config, Inkling2VisionConfig
from nemo_automodel.components.models.inkling2.multimodal import Inkling2ForConditionalGeneration
from nemo_automodel.components.models.inkling2.parallelization import FP32_MODULES
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from tests.unit_tests.models.inkling2.test_precision import initialize_nonuniform_queries
from tests.unit_tests.models.inkling2.test_reference import tiny_config


def dense(t: torch.Tensor) -> torch.Tensor:
    return (t.full_tensor() if isinstance(t, DTensor) else t).detach()


def worker(precision: str, output: Path) -> None:
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    try:
        bf16 = precision == "bf16"
        compute = torch.bfloat16 if bf16 else torch.float32
        torch.manual_seed(819)
        text = tiny_config()
        text.hidden_size = text.intermediate_size = 64
        text.n_routed_experts = 16
        text.vocab_size = text.unpadded_vocab_size = 200058
        cfg = Inkling2Config(
            architectures=["Inkling2ForConditionalGeneration"],
            text_config=text,
            vision_config=Inkling2VisionConfig(
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                mlp_ratio=2.0,
                posemb_grid_size=4,
                projection_hidden_dim=32,
            ),
        )
        backend = BackendConfig(
            attn="flex" if bf16 else "eager",
            linear="torch",
            rms_norm="torch",
            experts="torch_mm",
            dispatcher="torch",
            enable_fsdp_optimizations=True,
        )
        reference = Inkling2ForConditionalGeneration(copy.deepcopy(cfg), backend=copy.deepcopy(backend)).to(device)
        reference.visual.requires_grad_(False)
        initialize_nonuniform_queries(reference)
        initial = {k: t.detach().clone() for k, t in reference.state_dict().items()}

        def loss(logits, labels):
            return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")

        config = PipelineConfig(
            pp_schedule="interleaved1f1b",
            pp_batch_size=4,
            pp_microbatch_size=1,
            dtype=compute,
            pp_seq_len=9,
            loss_fn=loss,
            patch_stage_backward_maybe_with_nosync=True,
        )
        setup = DistributedSetup.build(
            strategy=FSDP2Config(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=compute, reduce_dtype=torch.float32, cast_forward_inputs=False
                ),
                reshard_after_forward=True,
            ),
            parallelism_sizes=ParallelismSizes(pp_size=2, ep_size=2, cp_size=2),
            pipeline_config=config,
            activation_checkpointing=True,
            world_size=4,
        )
        pipeline = NeMoAutoModelForImageTextToText.from_config(
            copy.deepcopy(cfg),
            dtype=torch.float32,
            backend=backend,
            attn_implementation="eager",
            use_liger_kernel=False,
            use_sdpa_patching=False,
            distributed_setup=setup,
            freeze_config={"freeze_vision_tower": True, "freeze_language_model": False},
        )
        mesh = setup.mesh_context
        lane = mesh.device_mesh["dp"].get_local_rank()
        parts = pipeline.parts
        for part in parts:
            set_model_state_dict(
                part, {k: initial[k] for k in part.state_dict()}, options=StateDictOptions(full_state_dict=True)
            )
            assert (part.visual is not None) == (part.embed_tokens is not None)
            if part.visual is not None:
                assert all(not p.requires_grad for p in part.visual.parameters())
        owners = {
            canonical_parameter_fqn(k): p for part in parts for k, p in part.named_parameters() if p.requires_grad
        }
        assert all(p.dtype == torch.float32 for p in owners.values())
        torch.manual_seed(820)  # Factory materialization consumes different RNG amounts on each stage.
        ids = (torch.arange(4 * 9, device=device).reshape(4, 9) % 35) + 1
        mask = torch.zeros_like(ids)
        counts = [0, 1, 2, 1]
        patchcounts = [0, 2, 3, 1]
        grids = [
            torch.tensor([[1, 1, 2]], device=device)
            if n == 2
            else torch.tensor([[1, 1, 1], [1, 1, 2]], device=device)
            if n == 3
            else torch.tensor([[1, 1, 1]], device=device)
            if n
            else torch.empty(0, 3, dtype=torch.long, device=device)
            for n in patchcounts
        ]
        patches = [torch.randn(n, 2, 40, 40, 3, device=device, dtype=torch.bfloat16) for n in patchcounts]
        import hashlib

        signature = hashlib.sha256(torch.cat(patches).cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        signatures = [None] * 4
        dist.all_gather_object(signatures, signature)
        assert len(set(signatures)) == 1, signatures
        for row, n in enumerate(patchcounts):
            ids[row, 5 : 5 + n] = 200005
            mask[row, 5 : 5 + n] = 1
        valid = torch.ones_like(ids, dtype=torch.bool)
        valid[0, 0] = False
        labels = (torch.arange(4 * 9, device=device).reshape(4, 9) % 35) + 1
        labels[mask.bool()] = -100
        labels[~valid] = -100
        labels[2] = -100
        nlabels = int((labels != -100).sum())
        rows = slice(lane * 4, (lane + 1) * 4)
        batch = prepare_vlm_media_for_pp(
            {
                "pixel_values": torch.cat(patches[rows]),
                "image_grid_thw": torch.cat(grids[rows]),
                "n_images_per_sample": torch.tensor(counts[rows], device=device),
            },
            batch_size=4,
            n_microbatches=4,
        )
        cpbatch = {
            "input_ids": ids[rows],
            "labels": labels[rows],
            "attention_mask": valid[rows],
            "mm_token_type_ids": mask[rows],
        }
        sharder = ContextParallelSharder(parts[0], mesh.device_mesh, cpbatch, num_chunks=4)
        context, cpbatch = sharder.shard(cpbatch)
        local_labels = cpbatch.pop("labels")
        local_ids = cpbatch.pop("input_ids")
        captured = []

        def scheduled_loss(logits, targets):
            captured.append(logits.detach())
            return loss(logits, targets)

        if pipeline.info.has_last_stage:
            pipeline.info.schedule._loss_fn = scheduled_loss
        losses = [] if pipeline.info.has_last_stage else None
        with stage_vlm_media_for_pp(pipeline, parts, batch):
            pipeline.step(
                local_ids, target=local_labels if pipeline.info.has_last_stage else None, losses=losses, **cpbatch
            )
        scale_grads_and_clip_grad_norm(
            None,
            parts,
            pp_enabled=True,
            device_mesh=mesh.device_mesh,
            pp_axis_name="pp",
            num_label_tokens=nlabels,
            dp_group_size=2,
            moe_mesh=mesh.moe_mesh,
            ep_axis_name="ep",
        )
        expected = [None] * 4
        for micro in range(4):
            effective = {
                k: p if any(pin in k for pin in FP32_MODULES) else p.to(compute)
                for k, p in reference.named_parameters()
            }
            group = [micro]
            media = {
                "pixel_values": torch.cat([patches[r] for r in group]),
                "image_grid_thw": torch.cat([grids[r] for r in group]),
                "mm_token_type_ids": mask[group],
                "attention_mask": valid[group],
            }
            logits = torch.func.functional_call(reference, effective, (ids[group],), media)
            for row, value in zip(group, logits.detach().split(1)):
                expected[row] = value
            loss(logits, labels[group]).backward()
        for p in reference.parameters():
            if p.grad is not None:
                p.grad.div_(nlabels)
        metrics = {
            "rank": rank,
            "bf16": bf16,
            "pp": 2,
            "ep": 2,
            "cp": 2,
            "ac": True,
            "schedule": "interleaved1f1b",
            "max_gradient_relative_l2": 0.0,
            "max_gradient_abs": 0.0,
        }
        if pipeline.info.has_last_stage:
            wanted = sharder.shard_token_tensor(torch.cat(expected[rows]), seq_dim=1, fill=0)
            keep = ~cpbatch["padding_mask"]
            err = (torch.cat(captured)[keep] - wanted[keep]).abs().max().item()
            metrics["max_logit_error"] = err
            assert err < (0.03 if bf16 else 2e-5), err
            error = abs(float(torch.stack(losses).sum()) - float(loss(wanted, local_labels))) / nlabels
            metrics["normalized_loss_error"] = error
            assert error < (0.02 if bf16 else 2e-5), error
        reference_parameters = dict(reference.named_parameters())
        for name, p in owners.items():
            assert p.grad is not None, name
            actual = dense(p.grad).float()
            wanted = reference_parameters[name].grad.float()
            error = actual - wanted
            relative = float(error.norm() / wanted.norm().clamp_min(1e-12))
            absolute = float(error.abs().max())
            metrics["max_gradient_relative_l2"] = max(relative, metrics["max_gradient_relative_l2"])
            metrics["max_gradient_abs"] = max(absolute, metrics["max_gradient_abs"])
            if bf16:
                assert relative < 0.1 or absolute < 5e-4, (name, relative, absolute)
            else:
                torch.testing.assert_close(actual, wanted, rtol=5e-5, atol=3e-6, msg=name)
            if name.endswith("relative_table"):
                assert actual.norm() > 0, name
        actualnorm = scale_grads_and_clip_grad_norm(
            0.25,
            parts,
            pp_enabled=True,
            device_mesh=mesh.device_mesh,
            pp_axis_name="pp",
            moe_mesh=mesh.moe_mesh,
            ep_axis_name="ep",
        )
        expectednorm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25, foreach=False)
        metrics["norm_relative_error"] = abs(float(actualnorm) - float(expectednorm)) / float(expectednorm)
        assert metrics["norm_relative_error"] < (0.03 if bf16 else 2e-4), metrics
        (output / f"image-pp-cp-{precision}-rank-{rank}.json").write_text(json.dumps(metrics, indent=2))
        if rank == 0:
            print("PASS image PP2/EP2/CP2 gradient and loss parity", precision, flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    worker(sys.argv[-2], Path(sys.argv[-1]))
