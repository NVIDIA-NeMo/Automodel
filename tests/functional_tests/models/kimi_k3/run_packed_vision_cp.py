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

"""Adversarial image packing with native collators and real KDA/MLA CP kernels.

FP32 checks every parameter gradient. BF16 uses the same rounded weights in an
FP32 precision control, separately for vision, language, and image-input gradients.
Individual BF16 gradient differences remain diagnostics: cancellation in a small
parameter tensor can amplify relative error without a corresponding Jacobian
error. The independent KDA recurrence and head-decay regression cover that path.
"""

import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import _build_packed_vlm_sample, _shift_sample
from nemo_automodel.components.models.kimi_k3.config import KimiK3Config
from nemo_automodel.components.models.kimi_k3.cp import KimiVlmBatch
from nemo_automodel.components.models.kimi_k3.model import KimiDeltaAttention
from nemo_automodel.components.models.kimi_k3.multimodal import KimiK3ForConditionalGeneration
from nemo_automodel.components.models.kimi_k3.vision import MoonViTEncoderLayer
from tests.unit_tests.models.kimi_k3.test_multimodal import _tiny_vlm


def make_model() -> KimiK3ForConditionalGeneration:
    """Create a four-layer CUDA fixture with head dimensions chosen to exercise KDA state handoffs."""
    torch.manual_seed(83)
    tiny = _tiny_vlm()
    cfg = tiny.config.to_dict()
    text = cfg["text_config"]
    text.update(
        hidden_size=128,
        head_dim=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        first_k_dense_replace=8,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        torch_dtype="bfloat16",
    )
    dtype = os.environ.get("K3_PARITY_DTYPE", "bfloat16")
    text["torch_dtype"] = dtype
    text["dtype"] = dtype
    text["linear_attn_config"].update(
        head_dim=int(os.environ.get("K3_KDA_HEAD_DIM", "64")), num_heads=2, kda_layers=[1, 2, 3], full_attn_layers=[4]
    )
    cfg["vision_config"].update(text_hidden_size=128)
    config = KimiK3Config(**cfg)
    config.vision_config._attn_implementation = "eager" if dtype == "float32" else "flash_attention_2"
    model = KimiK3ForConditionalGeneration(config, backend=tiny.backend).cuda()
    model.initialize_weights(torch.device("cuda"), dtype=getattr(torch, dtype))
    if os.environ.get("K3_RETAIN_CP_STATE") == "1":
        # Make cross-rank gradients survive more than one 64-token KDA chunk.
        # Fast-decaying random initializations cannot detect a wrong head's
        # decay in the backward state handoff because both states underflow.
        with torch.no_grad():
            for layer in model.model.layers.values():
                if isinstance(layer.self_attn, KimiDeltaAttention):
                    layer.self_attn.A_log.zero_()
                    layer.self_attn.dt_bias.view(layer.self_attn.num_heads, -1)[0].zero_()
                    layer.self_attn.dt_bias.view(layer.self_attn.num_heads, -1)[1:].fill_(-8.0)
    return model.train()


def make_samples(lengths: list[int], image_offset: int) -> list[dict[str, torch.Tensor]]:
    """Create unshifted image/text documents for adversarial packing.

    Args:
        lengths: Shifted token count for each of three documents.
        image_offset: Image-token start within the third document.

    Returns:
        Documents with token tensors [sequence], image patches [patches, 3, 2, 2],
        and image grids [images, 3] ordered time, height, width.
    """
    gen = torch.Generator().manual_seed(701)
    result = []
    # Unshifted documents contain one extra terminal target: packing shifts per document.
    for index, length in enumerate(lengths):
        ids = torch.randint(1, 60, (length + 1,), generator=gen)
        item = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
        if index == 0:
            ids[2:4] = 63
            item["pixel_values"] = torch.randn(8, 3, 2, 2, generator=gen)
            item["image_grid_thw"] = torch.tensor([[1, 4, 2]])
        if index == 2:
            ids[image_offset : image_offset + 8] = 63
            ids[-4:-3] = 63
            item["pixel_values"] = torch.randn(36, 3, 2, 2, generator=gen)
            item["image_grid_thw"] = torch.tensor([[1, 8, 4], [1, 2, 2]])
        item["labels"][ids == 63] = -100
        result.append(item)
    return result


def collate(
    samples: list[dict[str, torch.Tensor]], model: KimiK3ForConditionalGeneration, pack_size: int
) -> KimiVlmBatch:
    """Pack and collate the independently specified documents using native helpers.

    Args:
        samples: Documents with token tensors [sequence], image patches
            [patches, 3, 2, 2], and grids [images, 3].
        model: Packing contract owner.
        pack_size: Padded global token count.

    Returns:
        Batch with token tensors [1, pack_size], concatenated image patches
        [patches, 3, 2, 2], grids [images, 3], and cumulative document lengths.
    """
    sample = _build_packed_vlm_sample([_shift_sample(s) for s in samples], pack_size, 0)
    return neat_packed_vlm_collater([sample], padding_idx=0, max_length=pack_size, packing=model)


def on_device(batch: KimiVlmBatch) -> KimiVlmBatch:
    """Move a packed fixture to CUDA without changing its document layout.

    Args:
        batch: Token tensors [batch, sequence], pixels [patches, 3, 2, 2],
            grids [images, 3], and native packed document metadata.

    Returns:
        The same tensor layouts on CUDA, excluding labels and media-count vectors.
        FP32 validation promotes the already rounded pixels to FP32.
    """
    result = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
        if k not in ("labels", "n_images_per_sample", "n_videos_per_sample")
    }

    if os.environ.get("K3_PARITY_DTYPE") == "float32" and "pixel_values" in result:
        result["pixel_values"] = result["pixel_values"].float()
    return result


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Compare equally shaped tensors without reducing away large absolute errors.

    Args:
        actual: Tensor of arbitrary shape.
        expected: Reference tensor with the same shape.

    Returns:
        Maximum absolute error, relative RMS error, and 99th percentile absolute error.
    """
    actual, expected = actual.detach(), expected.detach()
    if actual.numel() == 0:
        return {"max_abs": 0.0, "rms_relative": 0.0, "p99_abs": 0.0}
    delta = actual.float() - expected.float()
    return {
        "max_abs": float(delta.abs().max()),
        "rms_relative": float(delta.square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-10)),
        "p99_abs": float(torch.quantile(delta.abs().flatten(), 0.99)),
    }


def main() -> None:
    """Exercise real CUDA attention and convolution across adversarial document boundaries."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Disable TORCH_ALLOW_TF32_CUBLAS_OVERRIDE before running precision parity.")
    if os.environ.get("K3_TEST_SAFE_GATE") == "0":
        from nemo_automodel.components.models.kimi_k3 import model as kda_module

        original_chunk = kda_module.chunk_kda

        def diagnostic_chunk(**kwargs):
            kwargs["safe_gate"] = False
            return original_chunk(**kwargs)

        kda_module.chunk_kda = diagnostic_chunk
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise ValueError("Run with TRITON_F32_DEFAULT=ieee; the default TF32 KDA path fails the FP64 gradient oracle.")
    reference = make_model()
    is_fp32 = os.environ.get("K3_PARITY_DTYPE") == "float32"
    precision_control = None
    if not is_fp32:
        precision_control = copy.deepcopy(reference).float()
        for layer in precision_control.vision_tower.modules():
            if isinstance(layer, MoonViTEncoderLayer):
                layer.attn_implementation = "eager"
    parallel = copy.deepcopy(reference)
    parallel.cp_mesh = mesh
    ref_stages = {}
    cp_stages = {}

    def recorder(store, name):
        def hook(module, args, output):
            if name not in store:
                tensor = output[0] if isinstance(output, tuple) else output
                store[name] = tensor.detach().clone()

        return hook

    for name, layer in reference.model.layers.items():
        layer.register_forward_hook(recorder(ref_stages, name))
    for name, layer in parallel.model.layers.items():
        layer.register_forward_hook(recorder(cp_stages, name))
    report = {
        "rank": rank,
        "kda_head_dim": int(os.environ.get("K3_KDA_HEAD_DIM", "64")),
        "retentive_cp_state": os.environ.get("K3_RETAIN_CP_STATE") == "1",
        "torch": torch.__version__,
        "dtype": os.environ.get("K3_PARITY_DTYPE", "bfloat16"),
        "cases": [],
    }
    report["kernel_variant"] = "safe_gate_false" if os.environ.get("K3_TEST_SAFE_GATE") == "0" else "production"
    report["kernel_variant"] = os.environ.get("K3_DIAGNOSTIC_KERNEL_LABEL", report["kernel_variant"])
    root = Path(os.environ["K3_RUN"])
    for case, lengths, pack_size, offset in [
        ("image_crosses_cp_cut", [61, 63, 151], 288, 16),
        ("doc_ends_before_cp_cut", [63, 80, 132], 256 + 32, 12),
        ("doc_starts_at_cp_cut", [63, 81, 129], 288, 1),
        ("doc_starts_after_cp_cut", [64, 81, 126], 288, 1),
        ("one_token_doc_at_cut", [143, 1, 131], 288, 1),
        ("unaligned_global_length", [61, 63, 151], 287, 16),
        ("entire_cp_rank_is_padding", [45, 31, 51], 288, 16),
    ]:
        ref_stages.clear()
        cp_stages.clear()
        samples = make_samples(lengths, offset)
        batch = collate(samples, reference, pack_size)
        ids = batch["attention_mask"]
        # Verify exact document ownership, per-document positions, and shifted labels.
        start = 0
        for document, sample in enumerate(samples, 1):
            length = lengths[document - 1]
            assert torch.equal(batch["labels"][0, start : start + length], sample["labels"][1:])
            assert batch["position_ids"][0, start : start + length].tolist() == list(range(length))
            assert (ids[0, start : start + length] == document).all()
            start += length
        assert (batch["labels"][0, start:] == -100).all()
        device_batch = on_device(batch)
        reference.zero_grad(set_to_none=True)
        pixels = device_batch["pixel_values"].detach().clone().requires_grad_()
        full = reference(**{**device_batch, "pixel_values": pixels}).logits
        full_precision = None
        precision_pixels = None
        if precision_control is not None:
            precision_control.zero_grad(set_to_none=True)
            precision_pixels = pixels.detach().float().requires_grad_()
            full_precision = precision_control(**{**device_batch, "pixel_values": precision_pixels}).logits
        separate = []
        for sample, length in zip(samples, lengths):
            single = on_device(collate([sample], reference, length))
            separate.append(reference(**single).logits)
        standalone = torch.cat(separate, 1)
        valid = sum(lengths)
        independent = metrics(full[:, :valid], standalone)
        torch.testing.assert_close(full[:, :valid], standalone, atol=0.02, rtol=0.03)
        cotangent = torch.randn(
            full.shape, device=full.device, generator=torch.Generator(device="cuda").manual_seed(912)
        ).float()
        cotangent[:, valid:] = 0
        (full.float() * cotangent).sum().div(valid).backward()
        expected_grads = {n: p.grad.detach().clone() for n, p in reference.named_parameters() if p.grad is not None}
        expected_pixel_grad = pixels.grad.detach().clone()
        precision_grads = {}
        precision_pixel_grad = None
        if precision_control is not None:
            (full_precision.float() * cotangent).sum().div(valid).backward()
            precision_grads = {
                name: parameter.grad.detach().clone()
                for name, parameter in precision_control.named_parameters()
                if parameter.grad is not None
            }
            precision_pixel_grad = precision_pixels.grad.detach().clone()
            full_precision = full_precision.detach()

        # Poison another document, keeping the target document and all shapes identical.
        poisoned = copy.deepcopy(samples)
        poisoned[0]["pixel_values"] *= -101
        poisoned[0]["input_ids"][0] = 59
        poison_batch = on_device(collate(poisoned, reference, pack_size))
        with torch.no_grad():
            perturbed = reference(**poison_batch).logits
        unaffected_start = lengths[0]
        torch.testing.assert_close(
            full[:, unaffected_start:valid], perturbed[:, unaffected_start:valid], atol=0, rtol=0
        )
        # Deliberately merge all documents; reconstruct metadata through the native collator.
        broken_samples = [_build_packed_vlm_sample([_shift_sample(s) for s in samples], pack_size, 0)]
        broken_samples[0]["attention_mask"].fill_(1)
        broken = on_device(neat_packed_vlm_collater(broken_samples, max_length=pack_size, packing=reference))
        broken_poison = {**broken, "pixel_values": poison_batch["pixel_values"]}
        with torch.no_grad():
            negative = (
                (
                    reference(**broken).logits[:, unaffected_start:valid]
                    - reference(**broken_poison).logits[:, unaffected_start:valid]
                )
                .abs()
                .max()
            )
        assert negative > 1e-4, (case, negative)

        parallel.zero_grad(set_to_none=True)
        cp_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        cp_pixels = cp_batch["pixel_values"].detach().clone().to(pixels.dtype).requires_grad_()
        cp_batch["pixel_values"] = cp_pixels
        sharder = parallel.prepare_model_inputs_for_cp(cp_batch)["cp_sharder"]
        _, cp_batch, layout = sharder.shard_batch(mesh, None, cp_batch)
        local_start = cp_batch["kimi_packed_seq_start"]
        cp_batch = {k: v for k, v in cp_batch.items() if k not in ("labels", "n_images_per_sample")}
        local = parallel(**cp_batch).logits
        assert torch.isfinite(local).all(), "Nonfinite logits on valid or padding positions"
        expected = full.detach()[:, local_start : local_start + local.shape[1]]
        count = min(local.shape[1], max(0, valid - local_start))
        logit_metrics = metrics(local[:, :count], expected[:, :count])
        stage_metrics = {
            name: metrics(value[:, :count], ref_stages[name][:, local_start : local_start + count])
            for name, value in cp_stages.items()
        }
        debug = {
            "case": case,
            "rank": rank,
            "valid_local_tokens": count,
            "logits": logit_metrics,
            "stages": stage_metrics,
            "padding": metrics(local[:, count : expected.shape[1]], expected[:, count:])
            if count < expected.shape[1]
            else None,
        }
        (root / f"packed_cp_debug_{report['dtype']}_rank{rank}.json").write_text(json.dumps(debug, indent=2))
        print("CP_STAGE_DEBUG", debug, flush=True)
        numeric_failures = []
        pairwise_gradient_diagnostics = []
        precision_metrics = None
        if is_fp32:
            torch.testing.assert_close(local[:, :count], expected[:, :count], atol=1e-4, rtol=1e-4)
        else:
            gold = full_precision[:, local_start : local_start + count]
            cp1_error = metrics(expected[:, :count], gold)
            cp2_error = metrics(local[:, :count], gold)
            precision_metrics = {"cp1_vs_same_weight_fp32": cp1_error, "cp2_vs_same_weight_fp32": cp2_error}
            # Both paths share the already-rounded BF16 weights and pixel values.
            # The FP32 graph and KDA FP64 recurrence are validated separately.
            if cp2_error["rms_relative"] > min(0.02, max(1.25 * cp1_error["rms_relative"], 1e-5)):
                numeric_failures.append("bf16_precision_control")
        global_positions = torch.arange(local_start, local_start + count, device=local.device)
        boundaries = [0, lengths[0], lengths[0] + lengths[1], valid]
        boundaries.extend(local.shape[1] * boundary for boundary in range(1, mesh.size()))
        near = torch.stack([(global_positions - boundary).abs() <= 3 for boundary in boundaries]).any(0)
        boundary_metrics = metrics(local[:, :count][:, near], expected[:, :count][:, near])
        if full_precision is not None:
            gold_boundary = full_precision[:, local_start : local_start + count][:, near]
            base_boundary = metrics(expected[:, :count][:, near], gold_boundary)
            cp_boundary = metrics(local[:, :count][:, near], gold_boundary)
            precision_metrics["boundary_cp1_vs_fp32"] = base_boundary
            precision_metrics["boundary_cp2_vs_fp32"] = cp_boundary
            if cp_boundary["rms_relative"] > max(1.5 * base_boundary["rms_relative"], 1e-5):
                numeric_failures.append("bf16_boundary_precision_control")
        padded_cotangent = torch.nn.functional.pad(
            cotangent, (0, 0, 0, local.shape[1] * mesh.size() - cotangent.shape[1])
        )
        (local.float() * padded_cotangent[:, local_start : local_start + local.shape[1]]).sum().div(valid).backward()
        gradient_metrics = {}
        gradient_precision = {}
        gradient_error_sums = {
            group: torch.zeros(3, dtype=torch.float64, device=local.device) for group in ("vision", "language")
        }
        for name, p in parallel.named_parameters():
            if name in expected_grads:
                assert p.grad is not None, name
                dist.all_reduce(p.grad)
                item = metrics(p.grad, expected_grads[name])
                gradient_metrics[name] = item
                if name in precision_grads:
                    gradient_precision[name] = {
                        "cp1_vs_same_weight_fp32": metrics(expected_grads[name], precision_grads[name]),
                        "cp_vs_same_weight_fp32": metrics(p.grad, precision_grads[name]),
                        "fp32_norm": float(precision_grads[name].float().norm()),
                    }
                    group = "vision" if name.startswith(("vision_tower.", "mm_projector.")) else "language"
                    gold_gradient = precision_grads[name].double()
                    gradient_error_sums[group] += torch.stack(
                        (
                            (expected_grads[name].double() - gold_gradient).square().sum(),
                            (p.grad.double() - gold_gradient).square().sum(),
                            gold_gradient.square().sum(),
                        )
                    )
                assert torch.isfinite(p.grad).all(), name
                if is_fp32 and item["rms_relative"] >= 1e-3:
                    numeric_failures.append(name)
                elif not is_fp32 and item["rms_relative"] >= 0.04:
                    pairwise_gradient_diagnostics.append(name)
        modality_gradient_precision = {}
        if not is_fp32:
            for group, sums in gradient_error_sums.items():
                cp1_error, cp_error = (sums[:2] / sums[2].clamp_min(1e-30)).sqrt().tolist()
                limit = min(0.05, max(1.25 * cp1_error, 1e-5))
                modality_gradient_precision[group] = {
                    "cp1_vs_same_weight_fp32_relative": cp1_error,
                    "cp_vs_same_weight_fp32_relative": cp_error,
                    "limit": limit,
                }
                if cp_error > limit:
                    numeric_failures.append(f"bf16_{group}_gradient_precision_control")
        dist.all_reduce(cp_pixels.grad)
        pixel_metrics = metrics(cp_pixels.grad, expected_pixel_grad)
        pixel_gradient_precision = None
        if precision_pixel_grad is not None:
            pixel_gradient_precision = {
                "cp1_vs_same_weight_fp32": metrics(expected_pixel_grad, precision_pixel_grad),
                "cp_vs_same_weight_fp32": metrics(cp_pixels.grad, precision_pixel_grad),
                "fp32_norm": float(precision_pixel_grad.float().norm()),
            }
        assert torch.isfinite(cp_pixels.grad).all()
        if is_fp32 and pixel_metrics["rms_relative"] >= 1e-3:
            numeric_failures.append("pixel_grad")
        elif not is_fp32:
            if pixel_metrics["rms_relative"] >= 0.02:
                pairwise_gradient_diagnostics.append("pixel_grad")
            cp1_error = pixel_gradient_precision["cp1_vs_same_weight_fp32"]["rms_relative"]
            cp_error = pixel_gradient_precision["cp_vs_same_weight_fp32"]["rms_relative"]
            limit = min(0.05, max(1.25 * cp1_error, 1e-5))
            pixel_gradient_precision["limit"] = limit
            if cp_error > limit:
                numeric_failures.append("bf16_pixel_gradient_precision_control")
        # Every CP rank receives the same poisoned full image stream, then slices
        # the embedded token stream through the production model-owned sharder.
        poisoned_cp = {**cp_batch, "input_ids": poison_batch["input_ids"], "pixel_values": poison_batch["pixel_values"]}
        with torch.no_grad():
            attacked = parallel(**poisoned_cp).logits
        unaffected = cp_batch["kimi_packed_doc_ids"][:, local_start : local_start + local.shape[1]] > 1
        torch.testing.assert_close(local.detach()[unaffected], attacked[unaffected], atol=0, rtol=0)
        parallel.zero_grad(set_to_none=True)
        cp_pixels.grad = None
        target_output = parallel(**cp_batch).logits
        target_mask = cp_batch["kimi_packed_doc_ids"][:, local_start : local_start + local.shape[1]] == 3
        (target_output.float() * target_mask.unsqueeze(-1)).sum().backward()
        dist.all_reduce(cp_pixels.grad)
        assert torch.count_nonzero(cp_pixels.grad[:8]) == 0, "cross-document image gradient leakage"
        assert torch.count_nonzero(cp_pixels.grad[8:]) > 0, "target image gradient was not exercised"
        entry = {
            "case": case,
            "lengths": lengths,
            "pack_size": pack_size,
            "cp_start": local_start,
            "input_sha256": hashlib.sha256(
                batch["input_ids"].numpy().tobytes() + batch["pixel_values"].float().numpy().tobytes()
            ).hexdigest(),
            "independent": independent,
            "cp_logits": logit_metrics,
            "pixel_grad": pixel_metrics,
            "parameter_grads": gradient_metrics,
            "parameter_gradient_precision_control": gradient_precision,
            "pixel_gradient_precision_control": pixel_gradient_precision,
            "negative_control_max": float(negative),
            "cp_other_document_output_exact": True,
            "cp_other_document_image_gradient_zero": True,
            "numeric_failures": numeric_failures,
            "legacy_pairwise_gradient_diagnostics": pairwise_gradient_diagnostics,
            "modality_gradient_precision_control": modality_gradient_precision,
            "precision_control": precision_metrics,
            "boundary_logits": boundary_metrics,
            "pixel_dtype": str(pixels.dtype),
            "triton_fp32_default": os.environ.get("TRITON_F32_DEFAULT", "default"),
            "cublas_tf32_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        report["cases"].append(entry)
        (root / f"packed_cp_{report['kernel_variant']}_{report['dtype']}_rank{rank}.json").write_text(
            json.dumps(report, indent=2)
        )
        print(case, independent, logit_metrics, pixel_metrics, flush=True)
    failures = {case["case"]: case["numeric_failures"] for case in report["cases"] if case["numeric_failures"]}
    dist.destroy_process_group()
    assert not failures, failures


if __name__ == "__main__":
    main()
