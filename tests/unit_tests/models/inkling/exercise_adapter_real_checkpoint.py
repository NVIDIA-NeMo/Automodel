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

"""Exercise the Inkling state-dict adapter against the *real* thinkingmachines/Inkling
checkpoint's full key inventory and tensor shapes -- without materializing the
975B weights.

Pipeline being validated (what a real ``from_pretrained`` does):

    raw checkpoint keys (model.llm.*, w13_weight, gate.bias, unembed, ...)
        --[ transformers conversion_mapping "inkling_mm_model" ]-->
    HF module keys (model.language_model.*.mlp.experts.gate_up_proj, ...)
        --[ InklingStateDictAdapter.from_hf ]-->
    native grouped keys (mlp.experts.gate_and_up_projs, ...)

Steps:
1. Pull the real ``config.json`` and the real safetensors key->shape/dtype map
   from the Hub (headers only, no bulk download).
2. Apply the real transformers conversion rules to the real (key, shape) list to
   obtain the HF-module (key, shape) map, and assert it matches an HF Inkling
   model built at the real config on ``meta``.
3. Run ``adapter.from_hf`` on that HF-module map (real shapes, ``meta`` tensors)
   and assert the result exactly covers the NeMo model's parameters (no missing /
   unexpected keys, matching shapes), plus a ``to_hf`` round-trip.

Run:  HF_HUB_OFFLINE=0 python tests/unit_tests/models/inkling/exercise_adapter_real_checkpoint.py
"""

import os
import re

import torch
from accelerate import init_empty_weights

REPO = "thinkingmachines/Inkling"
# Layers whose experts live under mtp are a separate multi-token-prediction head
# that neither HF nor NeMo instantiates; HF marks them ignore-on-load.
_MTP_RE = re.compile(r"(?:^|\.)mtp\.")


def _chunk(shape, dim, n=2):
    out = list(shape)
    out[dim] = out[dim] // n
    return out


def apply_transformers_conversion(real_shapes):
    """Apply the real ``inkling_mm_model`` conversion rules to a {key: shape} map.

    Uses transformers' own registered rule objects so this tracks upstream, but
    executes them at the shape level (Interleave = shape-preserving, Chunk = split).
    """
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming

    rules = get_checkpoint_conversion_mapping("inkling_mm_model")
    assert rules, "transformers has no inkling_mm_model conversion mapping"

    out = {}
    for key, shape in real_shapes.items():
        if _MTP_RE.search(key):
            continue  # ignore-on-load MTP head
        # Cumulatively apply every matching rule in order (a key may hit several
        # renamings plus one converter).
        current = {key: list(shape)}
        for rule in rules:
            srcs = rule.source_patterns
            srcs = srcs if isinstance(srcs, list) else [srcs]
            nxt = {}
            for k, s in current.items():
                matched = next((src for src in srcs if re.search(src, k)), None)
                if matched is None:
                    nxt[k] = s
                    continue
                if isinstance(rule, WeightRenaming):
                    tgt = rule.target_patterns
                    tgt = tgt[0] if isinstance(tgt, list) else tgt
                    nxt[re.sub(matched, tgt, k)] = s
                elif isinstance(rule, WeightConverter):
                    op_names = [type(op).__name__ for op in rule.operations]
                    tgts = rule.target_patterns
                    if "Chunk" in op_names:
                        chunk_dim = next(op.dim for op in rule.operations if type(op).__name__ == "Chunk")
                        piece = _chunk(s, chunk_dim, len(tgts))
                        for t in tgts:
                            nxt[re.sub(matched, t, k)] = list(piece)
                    else:
                        nxt[re.sub(matched, tgts[0], k)] = list(s)
            current = nxt
        out.update(current)
    return out


def build_real_config():
    os.environ.pop("HF_HUB_OFFLINE", None)
    from transformers.models.inkling.configuration_inkling import InklingConfig

    return InklingConfig.from_pretrained(REPO)


def main():
    from huggingface_hub import get_safetensors_metadata

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.inkling.model import InklingForConditionalGeneration

    os.environ.pop("HF_HUB_OFFLINE", None)

    # 1. Real config + real checkpoint key -> shape map (headers only).
    cfg = build_real_config()
    meta = get_safetensors_metadata(REPO)
    real_shapes = {}
    for fmeta in meta.files_metadata.values():
        for k, tinfo in fmeta.tensors.items():
            real_shapes[k] = list(tinfo.shape)
    print(f"[real] checkpoint tensors: {len(real_shapes)} (incl. MTP)")

    # 2. Real raw -> HF-module via transformers conversion rules.
    hf_shapes = apply_transformers_conversion(real_shapes)
    print(f"[conv] HF-module tensors after transformers conversion: {len(hf_shapes)}")

    # Authoritative HF-module params: build HF Inkling at the real config on meta.
    from transformers.models.inkling.modeling_inkling import (
        InklingForConditionalGeneration as HFModel,
    )

    with init_empty_weights():
        hf_model = HFModel(cfg)
    hf_param_shapes = {k: list(v.shape) for k, v in hf_model.state_dict().items()}

    # Every converted key must be a real HF param with the same shape.
    conv_missing = set(hf_param_shapes) - set(hf_shapes)
    conv_extra = set(hf_shapes) - set(hf_param_shapes)
    assert not conv_extra, f"conversion produced keys HF model does not have: {sorted(conv_extra)[:10]}"
    assert not conv_missing, f"HF params not produced by conversion: {sorted(conv_missing)[:10]}"
    for k in hf_shapes:
        assert hf_shapes[k] == hf_param_shapes[k], f"shape mismatch {k}: conv {hf_shapes[k]} vs hf {hf_param_shapes[k]}"
    print("[conv] real raw->HF-module conversion covers all HF params with matching shapes.")

    # 3. NeMo model at the real config on meta; run adapter on real HF-module shapes.
    with init_empty_weights():
        nemo_model = InklingForConditionalGeneration.from_config(cfg, backend=BackendConfig(experts="torch"))
    nemo_param_shapes = {k: list(v.shape) for k, v in nemo_model.state_dict().items()}

    hf_meta_sd = {k: torch.empty(s, device="meta") for k, s in hf_shapes.items()}
    native_sd = nemo_model.state_dict_adapter.from_hf(hf_meta_sd)
    native_shapes = {k: list(v.shape) for k, v in native_sd.items()}

    missing = set(nemo_param_shapes) - set(native_shapes)
    unexpected = set(native_shapes) - set(nemo_param_shapes)
    assert not missing, f"NeMo params NOT produced from real checkpoint: {sorted(missing)[:10]}"
    assert not unexpected, f"adapter produced keys NeMo model lacks: {sorted(unexpected)[:10]}"
    for k in native_shapes:
        assert native_shapes[k] == nemo_param_shapes[k], (
            f"shape mismatch {k}: adapter {native_shapes[k]} vs model {nemo_param_shapes[k]}"
        )
    print(f"[adapter] real checkpoint -> native covers all {len(nemo_param_shapes)} NeMo params, shapes match.")

    # Round-trip native -> HF keys.
    back = nemo_model.state_dict_adapter.to_hf(native_sd)
    assert set(back.keys()) == set(hf_shapes.keys()), "to_hf round-trip key mismatch"
    print("[adapter] to_hf round-trip reproduces the HF-module key set.")
    print("REAL-CHECKPOINT ADAPTER EXERCISE PASSED.")


if __name__ == "__main__":
    main()
