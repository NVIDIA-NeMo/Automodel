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

"""Packed Qwen3 with real TE attention and unsharded BF16 compute / FP32 parameters."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as HFQwen3ForCausalLM

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3.model import Qwen3ForCausalLM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and real TE attention")


@pytest.mark.parametrize("checkpointing", [False, True])
def test_qwen3_packed_single_rank_autocast(tmp_path: Path, checkpointing: bool) -> None:
    """Checkpoint-loaded FP32 Qwen3 must train packed documents with BF16 TE Q/K/V."""
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=256,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    checkpoint = tmp_path / "checkpoint"
    reference = HFQwen3ForCausalLM(config)
    with torch.no_grad():
        for layer in reference.model.layers:
            layer.self_attn.q_norm.weight.normal_(mean=1.0, std=0.25)
            layer.self_attn.k_norm.weight.normal_(mean=1.0, std=0.25)
    reference.save_pretrained(checkpoint)
    reference = reference.cuda().train()
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1, timeout=timedelta(seconds=120)
    )
    try:
        model = NeMoAutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            dtype=torch.float32,
            attn_implementation="sdpa",
            use_liger_kernel=False,
            use_sdpa_patching=False,
            backend=BackendConfig(attn="te", linear="torch", rms_norm="torch_fp32", rope_fusion=False),
            distributed_setup=DistributedSetup.build(
                strategy=FSDP2Config(), world_size=1, activation_checkpointing=checkpointing
            ),
        ).train()
        assert isinstance(model, Qwen3ForCausalLM)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        tokens = torch.randint(0, config.vocab_size, (64,), device="cuda")
        positions = torch.cat((torch.arange(16), torch.arange(48))).cuda()
        offsets = torch.tensor([0, 16, 64], dtype=torch.int32, device="cuda")
        logits = model(
            input_ids=tokens,
            position_ids=positions,
            qkv_format="thd",
            cu_seqlens=offsets,
            max_seqlen=48,
        ).logits
        assert logits.shape == (1, tokens.numel(), config.vocab_size)
        logits = logits.squeeze(0)
        # Independent, unpacked HF documents verify packing boundaries as well as numerics.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = torch.cat(
                [
                    reference(document.unsqueeze(0), use_cache=False).logits.squeeze(0)
                    for document in tokens.split((16, 48))
                ]
            )
        assert logits.dtype == torch.bfloat16
        torch.testing.assert_close(logits, expected, rtol=0.03, atol=0.005)
        upstream = torch.randn_like(logits) / logits.numel()
        logits.backward(upstream)
        expected.backward(upstream)
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            # Activation checkpoint wrappers retain the same underlying parameter names.
            reference_name = name.replace("._checkpoint_wrapped_module", "")
            assert parameter.dtype == torch.float32
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(
                parameter.grad, reference_parameters[reference_name].grad, rtol=0.05, atol=5e-5, msg=name
            )
            reference_gradient = reference_parameters[reference_name].grad
            relative_error = (parameter.grad - reference_gradient).norm() / reference_gradient.norm()
            assert relative_error < 0.03, (name, relative_error.item())
        original = model.model.layers[0].self_attn.q_norm.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        assert not torch.equal(original, model.model.layers[0].self_attn.q_norm.weight)
        assert all(
            state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32 for state in optimizer.state.values()
        )
    finally:
        dist.destroy_process_group()
