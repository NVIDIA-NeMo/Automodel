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

"""Compatibility patches for Phi-4 multimodal remote-code models."""


def _patch_phi4mm_processor() -> None:
    """Patch AutoProcessor.from_pretrained to fall back to Phi4MM remote code."""
    import transformers.processing_utils as pu

    if getattr(pu.ProcessorMixin.__dict__.get("from_pretrained"), "_nemo_phi4mm_patched", False):
        return
    _orig = pu.ProcessorMixin.from_pretrained.__func__

    @classmethod  # type: ignore[misc]
    def _patched(cls, pretrained_model_name_or_path, *args, **kwargs):
        try:
            return _orig(cls, pretrained_model_name_or_path, *args, **kwargs)
        except AttributeError as e:
            if "image_token" not in str(e) and "audio_token" not in str(e):
                raise
            import json

            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            kwargs.pop("trust_remote_code", None)
            repo = pretrained_model_name_or_path

            ProcessorCls = get_class_from_dynamic_module("processing_phi4mm.Phi4MMProcessor", repo)
            ImageProcCls = get_class_from_dynamic_module("processing_phi4mm.Phi4MMImageProcessor", repo)
            AudioProcCls = get_class_from_dynamic_module("processing_phi4mm.Phi4MMAudioFeatureExtractor", repo)

            pp_path = hf_hub_download(repo, "preprocessor_config.json")
            with open(pp_path) as f:
                pp_cfg = json.load(f)

            return ProcessorCls(
                ImageProcCls(dynamic_hd=pp_cfg.get("dynamic_hd", 36)),
                AudioProcCls(
                    audio_compression_rate=pp_cfg.get("audio_compression_rate", 8),
                    audio_downsample_rate=pp_cfg.get("audio_downsample_rate", 1),
                    audio_feat_stride=pp_cfg.get("audio_feat_stride", 1),
                ),
                AutoTokenizer.from_pretrained(repo, trust_remote_code=True),
            )

    _patched._nemo_phi4mm_patched = True  # type: ignore[attr-defined]
    pu.ProcessorMixin.from_pretrained = _patched


def apply_global_patches() -> None:
    """Apply process-wide compatibility patches for Phi4MM models."""
    _patch_phi4mm_processor()
