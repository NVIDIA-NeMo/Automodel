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

"""Multi-GPU parity check for EAGLE-3 target context parallelism.

Runs the frozen target's ``generate_batch`` under CP (cp_size == world_size) and
compares the gathered ``aux_hidden_states`` / ``logits`` against a non-CP
reference on identical weights and inputs. Also asserts CP does not corrupt the
caller's ``input_ids``. This is the correctness gate that no CPU/single-GPU unit
test can cover (the ring-attention forward + sequence gather only runs here).

Run on a node with >= 2 GPUs:

    torchrun --nproc_per_node=2 scripts/verify_eagle3_target_cp_parity.py

bf16 tolerances are used because ring attention reorders the float additions.
"""

from copy import deepcopy

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel


def _compare(a: torch.Tensor, b: torch.Tensor, name: str, rank: int) -> tuple[float, float]:
    max_diff = (a.float() - b.float()).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()
    print(f"[rank{rank}] {name}: shape={tuple(a.shape)} max_diff={max_diff:.3e} cos={cos:.6f}", flush=True)
    return max_diff, cos


def main() -> None:
    """Run the CP-vs-reference target parity check across the torchrun world."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    cp_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))["cp"]

    # Identical weights on every rank.
    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=512,
        max_position_embeddings=256,
        attn_implementation="sdpa",
    )
    base = LlamaForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()

    # Identical inputs on every rank; T divisible by world for a clean shard.
    torch.manual_seed(123)
    batch_size, seq_len = 1, 64
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    aux_ids = [1, 3, 5]
    # Separate copies: the CP wrapper attaches self_attn hooks that would
    # otherwise perturb the reference forward.
    ref = HFEagle3TargetModel(deepcopy(base), aux_layer_ids=aux_ids, cp_mesh=None)
    cp = HFEagle3TargetModel(deepcopy(base), aux_layer_ids=aux_ids, cp_mesh=cp_mesh)

    with torch.no_grad():
        b_ref = ref.generate_batch(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask)
        b_cp = cp.generate_batch(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask)

    md_aux, cs_aux = _compare(b_cp.aux_hidden_states, b_ref.aux_hidden_states, "aux_hidden_states", rank)
    md_log, cs_log = _compare(b_cp.logits, b_ref.logits, "logits", rank)
    input_ids_ok = torch.equal(b_cp.input_ids, b_ref.input_ids)
    print(f"[rank{rank}] input_ids_uncorrupted={input_ids_ok}", flush=True)

    ok = md_aux < 2e-2 and md_log < 2e-2 and cs_aux > 0.999 and cs_log > 0.999 and input_ids_ok
    print(f"[rank{rank}] PARITY {'PASS' if ok else 'FAIL'}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
