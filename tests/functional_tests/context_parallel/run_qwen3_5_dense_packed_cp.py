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

"""CP=2 versus CP=1 parity for NEAT-packed dense Qwen3.5 training.

Runs the production path: the model's CP hook selects block-diagonal CP for the
packed batch, the forward shards embeddings into contiguous chunks, full attention
runs per-document CP SDPA and the GatedDeltaNet resets its state at document
boundaries across ranks. Two of the three documents straddle the rank boundary.
A sensitivity check confirms that treating the pack as one document changes the
logits far more than CP does, so the parity tolerance detects boundary leaks.

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/context_parallel/run_qwen3_5_dense_packed_cp.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

DOC_LENGTHS = (19, 29, 16)
VOCAB = 128


def _config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )

    text_config = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        attn_implementation="sdpa",
        torch_dtype="bfloat16",
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=64,
    )
    return Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=VOCAB - 8,
        video_token_id=VOCAB - 7,
        vision_start_token_id=VOCAB - 6,
        vision_end_token_id=VOCAB - 5,
    )


def _model(device):
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration

    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", rope_fusion=False, dispatcher="torch")
    model = Qwen3_5ForConditionalGeneration(_config(), backend=backend, num_nextn_predict_layers=0)
    return model.to(device=device, dtype=torch.bfloat16).train()


def _packed_batch(contract, materialize_4d_mask, single_document=False):
    """Collate one NEAT pack of ``DOC_LENGTHS`` documents as the dataloader would.

    Args:
        contract: Packing contract returned by ``configure_packing_for_models``.
        materialize_4d_mask: False for CP, which rebuilds masks from document IDs.
        single_document: Pack the same tokens as one document (sensitivity check).

    Returns:
        Collated batch: ``input_ids``/``labels`` of shape [1, sequence], mRoPE
        ``position_ids`` of shape [3, 1, sequence], and packing metadata.
    """
    from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
    from nemo_automodel.components.datasets.vlm.neat_packing_vlm import _build_packed_vlm_sample

    generator = torch.Generator().manual_seed(4321)
    lengths = (sum(DOC_LENGTHS),) if single_document else DOC_LENGTHS
    tokens = torch.randint(1, VOCAB - 8, (sum(DOC_LENGTHS) + 1,), generator=generator)
    samples, start = [], 0
    for length in lengths:
        samples.append(
            {
                "input_ids": tokens[start : start + length].clone(),
                "labels": tokens[start + 1 : start + length + 1].clone(),
                "position_ids": torch.arange(length).expand(3, length).clone(),
            }
        )
        start += length
    pack = _build_packed_vlm_sample(samples, pack_size=sum(DOC_LENGTHS), padding_idx=0, has_mrope=True)
    return neat_packed_vlm_collater([pack], padding_idx=0, packing=contract, materialize_4d_mask=materialize_4d_mask)


def _model_inputs(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items() if key != "labels"
    }


def _sync(model):
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(tensor, src=0)


def main():
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"}.issubset(os.environ):
        print("ERROR: launch this script with torchrun.", file=sys.stderr)
        sys.exit(1)
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if world_size != 2:
        if rank == 0:
            print(f"ERROR: this parity test requires CP=2, got {world_size} ranks.", file=sys.stderr)
        dist.destroy_process_group()
        sys.exit(1)

    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.models.common.packing import configure_packing_for_models
    from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet

    torch.manual_seed(1234)
    reference = _model(device)
    distributed = _model(device)
    _sync(reference)
    distributed.load_state_dict(reference.state_dict())
    contract = configure_packing_for_models([reference])
    configure_packing_for_models([distributed])

    # CP=1 reference on the full pack, as a cp_size=1 run would see it.
    reference_batch = _packed_batch(contract, materialize_4d_mask=True)
    labels = reference_batch["labels"].to(device)
    ref_logits = reference(**_model_inputs(reference_batch, device)).logits
    valid = labels != -100
    ref_loss = F.cross_entropy(ref_logits.float()[valid], labels[valid])
    ref_loss.backward()

    # Sensitivity: the same tokens as one document must change the logits far more than CP does.
    with torch.no_grad():
        merged_logits = reference(**_model_inputs(_packed_batch(contract, True, single_document=True), device)).logits
    merged_diff = (merged_logits.float() - ref_logits.detach().float()).abs().max().item()

    # CP=2 through the model hook and its block-diagonal sharder, as the recipe runs it.
    cp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))["cp"]
    distributed.cp_mesh = cp_mesh
    for module in distributed.modules():
        if isinstance(module, CPAwareGatedDeltaNet):
            module._cp_mesh = cp_mesh
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in _packed_batch(contract, False).items()
    }
    prepared = distributed.prepare_model_inputs_for_cp(batch)
    sharder = prepared.pop("cp_sharder")
    batch.update(prepared)
    train_ctx, local_batch, layout = sharder.shard_batch(cp_mesh, None, batch)
    assert layout.padded_seq_len == layout.original_seq_len, "the pack length must not need CP padding"
    local_labels = local_batch.pop("labels")
    local_valid = local_labels != -100
    total_valid = local_valid.sum().clone()
    dist.all_reduce(total_valid)
    with train_ctx():
        local_logits = distributed(**local_batch).logits
        local_loss_sum = F.cross_entropy(local_logits.float()[local_valid], local_labels[local_valid], reduction="sum")
        (local_loss_sum / total_valid).backward()
    cp_loss = local_loss_sum.detach() / total_valid
    dist.all_reduce(cp_loss)
    gathered = [torch.empty_like(local_logits) for _ in range(world_size)]
    dist.all_gather(gathered, local_logits.detach().contiguous())
    cp_logits = torch.cat(gathered, dim=1)
    logit_diff = (cp_logits.float() - ref_logits.detach().float()).abs()

    diff_sq = ref_sq = cp_sq = dot = 0.0
    worst = []
    cp_parameters = dict(distributed.named_parameters())
    for name, ref_parameter in reference.named_parameters():
        cp_parameter = cp_parameters[name]
        if ref_parameter.grad is None:
            assert cp_parameter.grad is None, f"CP produced an unexpected gradient for {name}"
            continue
        assert cp_parameter.grad is not None, f"CP did not produce a gradient for {name}"
        cp_grad = cp_parameter.grad.detach().double().clone()
        dist.all_reduce(cp_grad)
        ref_grad = ref_parameter.grad.detach().double()
        assert torch.isfinite(cp_grad).all(), f"CP produced a non-finite gradient for {name}"
        diff_sq += (cp_grad - ref_grad).square().sum().item()
        ref_sq += ref_grad.square().sum().item()
        cp_sq += cp_grad.square().sum().item()
        dot += (cp_grad * ref_grad).sum().item()
        worst.append(((cp_grad - ref_grad).norm().item() / max(ref_grad.norm().item(), 1e-12), name))
    grad_rel_l2 = (diff_sq / ref_sq) ** 0.5
    grad_cosine = dot / max((cp_sq * ref_sq) ** 0.5, 1e-12)

    if rank == 0:
        print(
            f"Qwen3.5 dense packed CP: loss cp/ref={cp_loss.item():.6f}/{ref_loss.item():.6f}, "
            f"logits mean/max diff={logit_diff.mean().item():.3e}/{logit_diff.max().item():.3e} "
            f"(single-document max diff {merged_diff:.3e}), grad rel-L2/cosine={grad_rel_l2:.3e}/{grad_cosine:.6f}, "
            f"worst: {', '.join(f'{n}={v:.2e}' for v, n in sorted(worst, reverse=True)[:4])}",
            flush=True,
        )
    # bf16 with recurrent state carried rank-to-rank changes accumulation order versus CP=1.
    torch.testing.assert_close(cp_loss, ref_loss.detach(), rtol=2e-3, atol=2e-3)
    assert logit_diff.max().item() < 0.1 * merged_diff, (
        "CP logits are not closer to per-document attention than to one document"
    )
    assert grad_rel_l2 < 5e-2 and grad_cosine > 0.998, (
        f"CP gradient parity failed: {grad_rel_l2:.3e}, {grad_cosine:.6f}"
    )
    if rank == 0:
        print("Qwen3.5 dense packed CP parity PASS", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
