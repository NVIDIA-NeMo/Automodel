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

"""KL-divergence test for the custom Mixtral MoE model under expert parallelism.

Verifies that a 2-layer Mixtral MoE model with ``ep_size=2`` produces
logits consistent with an un-sharded (``ep_size=1``) reference.  Because
expert parallelism only partitions experts across ranks, the gathered
output should be numerically identical (or extremely close) to the output
produced without EP.

Launch with ``torch.distributed.run --nproc_per_node=2``.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from transformers.models.mixtral.configuration_mixtral import MixtralConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mixtral.model import MixtralForCausalLM
from nemo_automodel.components.moe.config import MoEParallelizerConfig
from nemo_automodel.components.moe.parallelizer import parallelize_model


def _kl_divergence_from_logits(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """KL(reference || candidate), per-token max."""
    assert reference_logits.shape == candidate_logits.shape
    vocab = reference_logits.shape[-1]
    ref_lp = F.log_softmax(reference_logits.float(), dim=-1).reshape(-1, vocab)
    cand_lp = F.log_softmax(candidate_logits.float(), dim=-1).reshape(-1, vocab)
    return F.kl_div(cand_lp, ref_lp, reduction="none", log_target=True).sum(-1)


def _build_config() -> MixtralConfig:
    return MixtralConfig(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        num_local_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.0,
        router_jitter_noise=0.0,
        use_cache=False,
    )


def _build_backend() -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=True,
        enable_hf_state_dict_adapter=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kl-threshold", type=float, default=1e-4)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    config = _build_config()
    backend = _build_backend()

    # ── Reference run: ep_size=1 (all experts on every rank) ──────────
    torch.manual_seed(42)
    ref_model = MixtralForCausalLM(config, backend=backend).to(device).to(torch.bfloat16)
    ref_state = {k: v.clone() for k, v in ref_model.state_dict().items()}

    torch.manual_seed(123)
    input_ids = torch.randint(0, config.vocab_size, (2, 64), device=device)

    with torch.no_grad():
        ref_logits = ref_model(input_ids=input_ids).float()

    del ref_model
    torch.cuda.empty_cache()

    # ── EP run: ep_size=2 (experts sharded across 2 ranks) ───────────
    torch.manual_seed(42)
    ep_model = MixtralForCausalLM(config, backend=backend).to(device).to(torch.bfloat16)
    ep_model.load_state_dict(ref_state, strict=True)

    ep_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ep",))
    moe_parallelizer_config = MoEParallelizerConfig()
    parallelize_model(ep_model, ep_mesh, moe_parallelizer_config)

    torch.manual_seed(123)
    input_ids_ep = torch.randint(0, config.vocab_size, (2, 64), device=device)

    with torch.no_grad():
        ep_logits = ep_model(input_ids=input_ids_ep).float()

    # ── Compare ───────────────────────────────────────────────────────
    kl_per_token = _kl_divergence_from_logits(ref_logits, ep_logits)
    max_kl = kl_per_token.max().item()

    if rank == 0:
        status = "PASS" if max_kl < args.kl_threshold else "FAIL"
        print(f"{status}: Mixtral EP KL-divergence test  (max_kl={max_kl:.6g}, threshold={args.kl_threshold:g})")

    dist.destroy_process_group()

    if max_kl >= args.kl_threshold:
        sys.exit(1)


if __name__ == "__main__":
    main()
