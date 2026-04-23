# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Devstral-Small-2-24B and Devstral-2-123B FP8 checkpoint loaders.

HF's ``FineGrainedFP8Config(dequantize=True)`` loader inside
``transformers.PreTrainedModel.from_pretrained`` materializes the full
dequantized BF16 model on every rank *before* PP split, OOM-ing on 80 GB H100
at TP=2 PP=2 (~77 GB/rank). We avoid that by:

  1. Constructing the model on meta via the standard nemo_automodel
     custom-model path — no GPU allocation, no HF quant loader.
  2. Attaching a ``state_dict_adapter`` (see ./state_dict_adapter.py,
     modelled on ``deepseek_v3/state_dict_adapter.py``) that handles both
     (a) key remapping between HF's on-disk layout and our text-only layout,
     and (b) FP8 → BF16 dequantization during the standard
     ``Checkpointer.load_model`` DCP load.

Routing: ``__init__.py`` installs a resolver hook on
``_resolve_custom_model_cls_for_config`` so Devstral FP8 configs dispatch
to these classes *without* overwriting the existing Mistral3 / Mistral4
registry entries (non-FP8 users keep the stock path).
"""

from __future__ import annotations

import logging

import torch
from transformers import PretrainedConfig

from nemo_automodel.components.models.devstral.state_dict_adapter import (
    DevstralFP8StateDictAdapter,
)
from nemo_automodel.components.models.mistral3.model import (
    Ministral3Config,
    Ministral3ForCausalLM,
)

logger = logging.getLogger(__name__)


class Devstral24BFP8TextForCausalLM(Ministral3ForCausalLM):
    """Text-only Devstral-Small-2-24B (FP8 VLM checkpoint).

    The underlying HF architecture is ``Mistral3ForConditionalGeneration``
    (a VLM wrapping a ``ministral3`` text decoder). We only train the text
    stack, so the config is unwrapped to its inner ``Ministral3Config`` and
    the state-dict adapter strips the ``language_model.`` prefix during load.
    """

    # See checkpointing.py:initialize_model_weights — gate on this class
    # attribute to skip HF's `initialize_weights()` since the state-dict
    # adapter path will populate every tensor anyway. Skipping also avoids
    # a stage-divergent DTensor collective inside HF's init on PP setups.
    _skip_init_weights_on_load = True

    def __init__(self, config: PretrainedConfig):
        ministral_cfg = self._coerce_to_ministral_config(config)
        super().__init__(ministral_cfg)
        # The adapter runs inside Checkpointer.load_model via
        # _maybe_adapt_state_dict_{to,from}_hf.  24B VLM keys all carry the
        # `language_model.` prefix on disk.
        self.state_dict_adapter = DevstralFP8StateDictAdapter(key_prefix="language_model.")

    @staticmethod
    def _coerce_to_ministral_config(config: PretrainedConfig) -> Ministral3Config:
        """Return a Ministral3Config; unwrap `text_config` if a VLM config was passed.

        We deliberately keep ``quantization_config`` on the returned config so
        that ``apply_model_infrastructure`` (infrastructure.py:419-422) sets
        ``dequantize_base_checkpoint=True`` on the Checkpointer, which in turn
        causes ``_maybe_adapt_state_dict_to_hf`` to call ``adapter.to_hf`` with
        ``quantization=True`` — that's how our adapter knows to emit the
        ``_scale_inv`` placeholder keys that DCP will populate from disk.
        HF's ``from_pretrained`` is never invoked for this class (custom
        model path), so the lingering ``quantization_config`` never reaches
        HF's FineGrainedFP8 loader.
        """
        if isinstance(config, Ministral3Config):
            return config
        text_config = getattr(config, "text_config", None)
        if text_config is None:
            raise TypeError(
                f"{Devstral24BFP8TextForCausalLM.__name__} expected a Ministral3Config "
                f"or a config with .text_config, got {type(config).__name__}."
            )
        # Copy the outer VLM's quantization_config dict onto the text config —
        # otherwise the Ministral3Config constructor drops it (it's not part of
        # the ministral3 schema). Setting it post-init keeps `hasattr` True,
        # which is all apply_model_infrastructure checks.
        ministral_cfg = Ministral3Config(**text_config.to_dict())
        ministral_cfg.name_or_path = getattr(config, "name_or_path", "") or getattr(
            config, "_name_or_path", ""
        )
        outer_qc = getattr(config, "quantization_config", None)
        if outer_qc is not None:
            ministral_cfg.quantization_config = outer_qc
        return ministral_cfg

    @classmethod
    def supports_config(cls, config: PretrainedConfig) -> bool:
        """Activate only on FP8-native Mistral3 VLMs (Devstral-Small-2-24B)."""
        text_config = getattr(config, "text_config", None)
        if text_config is None or getattr(text_config, "model_type", None) != "ministral3":
            return False
        qc = getattr(config, "quantization_config", None)
        if qc is None:
            return False
        method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
        return method == "fp8"


class Devstral123BFP8ForCausalLM(Ministral3ForCausalLM):
    """Dense text-only Devstral-2-123B (FP8 Ministral3 checkpoint).

    Differences vs the 24B variant:
      - Underlying arch is ``Ministral3ForCausalLM`` (not a VLM) — no prefix
        to strip, no vision modules.
      - Config is a ``Ministral3Config`` directly; no ``text_config`` unwrap.
    """

    _skip_init_weights_on_load = True

    def __init__(self, config: PretrainedConfig):
        if not isinstance(config, Ministral3Config):
            if getattr(config, "model_type", None) != "ministral3":
                raise TypeError(
                    f"{type(self).__name__} expects a ministral3 config, got "
                    f"{type(config).__name__} with model_type={getattr(config, 'model_type', None)!r}."
                )
            # Preserve quantization_config across the Ministral3Config coercion —
            # see Devstral24BFP8TextForCausalLM._coerce_to_ministral_config for
            # the rationale (apply_model_infrastructure uses it as a signal).
            outer_qc = getattr(config, "quantization_config", None)
            config = Ministral3Config(**config.to_dict())
            if outer_qc is not None:
                config.quantization_config = outer_qc
        super().__init__(config)
        # Pass-through on keys; only the dequant part of the adapter is active.
        self.state_dict_adapter = DevstralFP8StateDictAdapter(key_prefix="")

    @classmethod
    def supports_config(cls, config: PretrainedConfig) -> bool:
        """Activate only on FP8-native Ministral3 configs (Devstral-2-123B)."""
        if getattr(config, "model_type", None) != "ministral3":
            return False
        qc = getattr(config, "quantization_config", None)
        if qc is None:
            return False
        method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
        return method == "fp8"
