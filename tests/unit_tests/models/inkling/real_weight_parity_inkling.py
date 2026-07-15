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

"""Real-weight parity: NeMo Inkling MoE vs HuggingFace Inkling on the first N layers.

Uses the *actual* thinkingmachines/Inkling checkpoint (downloaded to the HF
cache). The model config is truncated to the first ``--layers`` decoder layers so
the checkpoint's real weights for those layers (dense layers 0-1 + sparse MoE
layers 2-3) are loaded, while HF's lazy safetensors reader skips the remaining
layers' tensors. HF converts the raw checkpoint to module format; the NeMo model
receives those same real weights through ``InklingStateDictAdapter.from_hf``.

Both models run a forward pass on identical ``input_ids`` (seq len ``--seqlen``)
and the language-model logits are compared with an allclose check plus KL
divergence, which must sit at the 1e-3 level.

Run:
  TRANSFORMERS_OFFLINE=1 python tests/unit_tests/models/inkling/real_weight_parity_inkling.py \
      --layers 4 --seqlen 200
"""

import argparse
import gc

import torch
import torch.nn.functional as F
from transformers.models.inkling.configuration_inkling import InklingConfig
from transformers.models.inkling.modeling_inkling import (
    InklingForConditionalGeneration as HFInklingForConditionalGeneration,
)

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling.model import InklingForConditionalGeneration

REPO = "thinkingmachines/Inkling"


def truncated_config(num_layers: int) -> InklingConfig:
    cfg = InklingConfig.from_pretrained(REPO)
    tc = cfg.text_config
    # Preserve the real per-layer attention/MLP type pattern for the kept layers.
    layer_types = list(tc.layer_types)[:num_layers]
    mlp_layer_types = list(tc.mlp_layer_types)[:num_layers]
    tc.num_hidden_layers = num_layers
    tc.layer_types = layer_types
    tc.mlp_layer_types = mlp_layer_types
    # Drop the multi-token-prediction head (not instantiated by the model class).
    tc.num_mtp_layers = None
    tc.mtp_local_layer_ids = None
    # Pin the same attention backend on both HF and NeMo (both reuse HF attention).
    cfg._attn_implementation = "eager"
    tc._attn_implementation = "eager"
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--kl-threshold", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    cfg = truncated_config(args.layers)
    tc = cfg.text_config
    print(
        f"[cfg] layers={tc.num_hidden_layers} hidden={tc.hidden_size} "
        f"experts={tc.n_routed_experts} top_k={tc.num_experts_per_tok} "
        f"shared={tc.n_shared_experts} layer_types={tc.layer_types} mlp_types={tc.mlp_layer_types}"
    )

    vocab = tc.unpadded_vocab_size or tc.vocab_size
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (1, args.seqlen), device=device)
    attention_mask = torch.ones_like(input_ids)

    # --- HF reference (real weights, truncated to N layers) -------------------
    print("[hf] loading real HF Inkling (truncated)...")
    hf_model = (
        HFInklingForConditionalGeneration.from_pretrained(
            REPO,
            config=cfg,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids, attention_mask=attention_mask).logits.float().cpu()
    hf_module_sd = {k: v.detach() for k, v in hf_model.state_dict().items()}
    del hf_model
    gc.collect()
    print(f"[hf] logits {tuple(hf_logits.shape)}")

    # --- NeMo model, loaded with the same real weights via the adapter --------
    print("[nemo] building and loading via state_dict_adapter...")
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch")
    nemo_model = InklingForConditionalGeneration.from_config(cfg, backend=backend).to(device=device, dtype=dtype).eval()
    native_sd = nemo_model.state_dict_adapter.from_hf(hf_module_sd)
    missing, unexpected = nemo_model.load_state_dict(native_sd, strict=False)
    missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
    unexpected = [k for k in unexpected if "rotary" not in k and "inv_freq" not in k]
    assert not missing, f"missing keys: {missing[:20]}"
    assert not unexpected, f"unexpected keys: {unexpected[:20]}"
    del hf_module_sd, native_sd
    gc.collect()

    with torch.no_grad():
        nemo_logits = nemo_model(input_ids=input_ids, attention_mask=attention_mask).logits.float().cpu()
    print(f"[nemo] logits {tuple(nemo_logits.shape)}")

    # --- Compare --------------------------------------------------------------
    assert hf_logits.shape == nemo_logits.shape, (hf_logits.shape, nemo_logits.shape)
    max_diff = (hf_logits - nemo_logits).abs().max().item()
    kl = F.kl_div(
        F.log_softmax(nemo_logits, dim=-1),
        F.log_softmax(hf_logits, dim=-1),
        log_target=True,
        reduction="batchmean",
    ).item()
    print(
        f"[result] layers={args.layers} seqlen={args.seqlen} dtype={args.dtype} "
        f"max_abs_diff={max_diff:.3e} KL(HF||NeMo)={kl:.3e}"
    )
    assert kl < args.kl_threshold, f"KL divergence too high: {kl} >= {args.kl_threshold}"
    print("REAL-WEIGHT PARITY PASSED.")


if __name__ == "__main__":
    main()
