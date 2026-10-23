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

"""Export a trained Automodel EAGLE-4 draft in the ``LlamaForCausalLMEagle4`` serving layout.

The EAGLE-4 loaders (TokenSpeed's ``LlamaForCausalLMEagle4`` and the EAGLE-4
engine plugins) refuse unknown config keys and checkpoint tensors that reach no
parameter, while an Automodel checkpoint carries the training-side config (the
target's config plus recipe keys) and wraps the backbone as ``model.``. The
export writes:

* ``config.json``: ``architectures=["LlamaForCausalLMEagle4"]``,
  ``model_type="llama"``, ``num_hidden_layers=1`` and only the draft-structure
  keys (sizes, norms, RoPE, sliding window, aux-layer ids, candidate head).
* ``model.safetensors``: every tensor with the ``model.`` prefix dropped
  (``fc.weight``, ``layers.0.*``, ``norm.weight``, ``embed_tokens.weight``) next
  to the top-level ``lm_head.weight``, ``efficient_lm_head.weight`` and
  ``efficient_lm_head_down_proj.weight``. ``embed_tokens`` is the target's
  (the recipe keeps it frozen), which the serving adapters load in its place.

Usage::

    python -m nemo_automodel.components.speculative.eagle.export_eagle4 \\
        <checkpoint>/model/consolidated --output-dir <export_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from nemo_automodel.components.speculative.eagle.draft_llama import EAGLE4_ARCHITECTURE
from nemo_automodel.components.speculative.eagle.eagle4 import CANDIDATE_HEAD_KEYS

logger = logging.getLogger(__name__)

_MODEL_PREFIX = "model."
# Draft-structure keys copied from the training config when present.
_COPIED_CONFIG_KEYS = (
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "eagle_config",
    "efficient_lm_head_config",
    "eos_token_id",
    "fc_norm",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "mlp_bias",
    "norm_output",
    "num_attention_heads",
    "num_key_value_heads",
    "pad_token_id",
    "rms_norm_eps",
    "rope_parameters",
    "sliding_window",
    "use_sliding_window",
    "vocab_size",
)
_LAYER_KEYS = (
    "hidden_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _resolve_checkpoint_dir(path: Path) -> Path:
    """Return the directory holding the draft ``config.json`` (``path`` or its ``model/consolidated``)."""
    for candidate in (path, path / "model" / "consolidated"):
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"No config.json in {path} or {path / 'model' / 'consolidated'}; pass the consolidated "
        "checkpoint directory written with checkpoint.save_consolidated."
    )


def _serving_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a training-side draft config and return the serving ``config.json``.

    Raises:
        ValueError: if the checkpoint is not a complete EAGLE-4 draft or uses a
            structure the EAGLE-4 loaders do not build.
    """
    if EAGLE4_ARCHITECTURE not in (config.get("architectures") or ()):
        raise ValueError(
            f"Not an EAGLE-4 draft (architectures={config.get('architectures')!r}); train it with recipe_args.eagle4."
        )
    if not config.get("efficient_lm_head_config"):
        raise ValueError(
            "The checkpoint has no candidate head; train it with recipe_args.eagle4.candidate_head "
            "(stage 2) before exporting."
        )
    vocab_size = config["vocab_size"]
    if config.get("draft_vocab_size", vocab_size) != vocab_size:
        raise ValueError(f"EAGLE-4 drafts use the full vocabulary; got draft_vocab_size={config['draft_vocab_size']}.")
    if config.get("parallel_drafting", False):
        raise ValueError("parallel_drafting (P-EAGLE) drafts cannot be exported as EAGLE-4.")
    if config.get("rope_parameters") is None:
        raise ValueError("The draft config has no rope_parameters; the serving engine would assume default RoPE.")
    if config.get("num_aux_hidden_states", 3) != 3:
        raise ValueError("EAGLE-4 serving engines take exactly three aux hidden states.")
    if config.get("target_hidden_size", config["hidden_size"]) != config["hidden_size"]:
        raise ValueError("EAGLE-4 serving engines need target_hidden_size == hidden_size.")
    if config.get("attention_bias", False) or config.get("mlp_bias", False) or config.get("hidden_act") != "silu":
        raise ValueError(
            "EAGLE-4 serving engines build a bias-free SiLU draft layer; got "
            f"attention_bias={config.get('attention_bias')}, mlp_bias={config.get('mlp_bias')}, "
            f"hidden_act={config.get('hidden_act')!r}."
        )

    serving = {key: config[key] for key in _COPIED_CONFIG_KEYS if key in config}
    serving.update(
        architectures=[EAGLE4_ARCHITECTURE],
        model_type="llama",
        draft_vocab_size=vocab_size,
        head_dim=config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"],
        num_aux_hidden_states=3,
        num_hidden_layers=1,
        target_hidden_size=config["hidden_size"],
        tie_word_embeddings=False,
    )
    return serving


def _serving_state_dict(state_dict: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Drop the ``model.`` prefix and check the tensors match the EAGLE-4 draft exactly.

    Args:
        state_dict: Mapping from Automodel draft key to tensor; tensor shapes are
            those of ``LlamaEagle3DraftModel`` and are left unchanged.
        config: The training-side draft config.

    Returns:
        Mapping from serving key to the same tensors (no copies).
    """
    renamed = {key.removeprefix(_MODEL_PREFIX): tensor for key, tensor in state_dict.items()}
    expected = {
        "embed_tokens.weight",
        "fc.weight",
        "norm.weight",
        "lm_head.weight",
        *CANDIDATE_HEAD_KEYS,
        *(f"layers.0.{key}" for key in _LAYER_KEYS),
    }
    if config.get("fc_norm", False):
        expected.update(f"fc_norm.{i}.weight" for i in range(3))
    missing, unexpected = sorted(expected - renamed.keys()), sorted(renamed.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint tensors do not match an EAGLE-4 draft: missing {missing}, unexpected {unexpected}."
        )
    return renamed


def export_eagle4_draft(checkpoint_dir: str | Path, output_dir: str | Path) -> Path:
    """Write the serving ``config.json`` and ``model.safetensors`` of an Automodel EAGLE-4 draft.

    Args:
        checkpoint_dir: The consolidated checkpoint directory (or the step
            directory that contains ``model/consolidated``).
        output_dir: Where to write the export; must differ from the checkpoint.

    Returns:
        The output directory.
    """
    source = _resolve_checkpoint_dir(Path(checkpoint_dir))
    output = Path(output_dir)
    if output.resolve() == source.resolve():
        raise ValueError("output_dir must differ from the checkpoint directory.")
    config = json.loads((source / "config.json").read_text())
    serving_config = _serving_config(config)  # before loading any weights
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files in {source}.")
    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        state_dict.update(load_file(str(shard)))

    tensors = _serving_state_dict(state_dict, config)
    serving_config["torch_dtype"] = str(tensors["lm_head.weight"].dtype).removeprefix("torch.")
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "model.safetensors"), metadata={"format": "pt"})
    (output / "config.json").write_text(json.dumps(serving_config, indent=2, sort_keys=True) + "\n")
    logger.info("Exported EAGLE-4 draft (%d tensors) to %s", len(tensors), output)
    return output


def main(argv: list[str] | None = None) -> None:
    """CLI entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        description="Export an Automodel EAGLE-4 draft in the LlamaForCausalLMEagle4 layout."
    )
    parser.add_argument("checkpoint_dir", help="Consolidated Automodel EAGLE-4 checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to write config.json and model.safetensors.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    export_eagle4_draft(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
