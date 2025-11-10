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

import gc
import logging
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import TRANSFORMERS_CACHE, AutoModelForCausalLM, AutoTokenizer

from nemo_automodel.recipes.llm.benchmark import BenchmarkingRecipeForNextTokenPrediction

logger = logging.getLogger(__name__)


def deepseek_v3_gate_process_fn(hf_tuple, auto_tuple) -> torch.Tensor:
    # Reorder HF tuple: (indices, weights) and sort by indices
    sorted_hf_indices, sorted_indices = hf_tuple[0].sort(dim=-1)
    sorted_hf_weights = hf_tuple[1].gather(dim=-1, index=sorted_indices)
    hf_tuple = (sorted_hf_indices, sorted_hf_weights)

    # Reorder Automodel tuple: (weights, indices) -> (indices, weights) and sort by indices
    auto_tuple = (auto_tuple[1], auto_tuple[0])  # Swap to (indices, weights)
    sorted_auto_indices, sorted_indices = auto_tuple[0].sort(dim=-1)
    sorted_auto_weights = auto_tuple[1].gather(dim=-1, index=sorted_indices)
    auto_tuple = (sorted_auto_indices, sorted_auto_weights)

    return hf_tuple, auto_tuple


GATE_PROCESS_FUNCTIONS = {
    "DeepseekV3ForCausalLM": deepseek_v3_gate_process_fn,
}
# ============================================================================
# Activation Recording
# ============================================================================


class ActivationRecorder:
    def __init__(self):
        self.activations: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _tensorize(self, output) -> Optional[torch.Tensor]:
        if output is None:
            return None
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)) and len(output) > 0:
            # For tuples/lists, try to extract tensors
            tensors = []
            for item in output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
            if len(tensors) == 1:
                return tensors[0]
            elif len(tensors) > 1:
                return tuple(tensors)
        return None

    def register(self, model: nn.Module) -> None:
        for module_name, module in model.named_modules():
            # Skip the root module (empty name)
            if not module_name:
                continue

            def _make_hook(name: str):
                def _hook(module_ref, input_ref, output):
                    # module_ref and input_ref are unused but required by the hook signature
                    del module_ref, input_ref  # Explicitly mark as intentionally unused

                    tensor = self._tensorize(output)
                    if tensor is None:
                        return

                    # Handle tuple outputs
                    if isinstance(tensor, tuple):
                        self.activations[name] = tuple(t.detach().cpu() for t in tensor)
                    else:
                        # Move to CPU immediately to save GPU memory
                        self.activations[name] = tensor.detach().cpu()

                return _hook

            self._handles.append(module.register_forward_hook(_make_hook(module_name)))

    def clear(self):
        self.activations.clear()

    def remove(self):
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()


# ============================================================================
# Comparison Utilities
# ============================================================================


def compare_and_print_logits(
    automodel_logits: torch.Tensor,
    hf_logits: torch.Tensor,
    tolerances: Sequence[Tuple[float, float, str]] = (
        (1e-4, 1e-6, "Very Strict"),
        (1e-2, 1e-4, "Strict"),
        (1e-1, 1e-2, "Moderate"),
    ),
) -> None:
    # Apply log_softmax to convert logits to log probabilities
    hf_logits = torch.log_softmax(hf_logits.float(), dim=-1, dtype=torch.float32)
    automodel_logits = torch.log_softmax(automodel_logits.float(), dim=-1, dtype=torch.float32)

    diff = torch.abs(automodel_logits - hf_logits)
    max_abs = float(torch.max(diff))
    mean_abs = float(torch.mean(diff))
    median_abs = float(torch.median(diff))
    rel = diff / (torch.abs(automodel_logits) + 1e-8)
    max_rel = float(torch.max(rel))
    mean_rel = float(torch.mean(rel))

    # Compute token-wise KL divergence in both directions
    # KL(P || Q) = sum(P * (log(P) - log(Q)))
    # Since we have log probabilities, KL(P || Q) = sum(exp(log_P) * (log_P - log_Q))

    # Convert log probs to probs
    automodel_probs = torch.exp(automodel_logits)
    hf_probs = torch.exp(hf_logits)

    # KL(automodel || hf) - measures how much information is lost when using hf to approximate automodel
    kl_auto_to_hf = (automodel_probs * (automodel_logits - hf_logits)).sum(dim=-1)

    # KL(hf || automodel) - measures how much information is lost when using automodel to approximate hf
    kl_hf_to_auto = (hf_probs * (hf_logits - automodel_logits)).sum(dim=-1)

    # Statistics on token-wise KL divergences
    kl_auto_to_hf_mean = float(kl_auto_to_hf.mean())
    kl_auto_to_hf_median = float(kl_auto_to_hf.median())
    kl_auto_to_hf_max = float(kl_auto_to_hf.max())

    kl_hf_to_auto_mean = float(kl_hf_to_auto.mean())
    kl_hf_to_auto_median = float(kl_hf_to_auto.median())
    kl_hf_to_auto_max = float(kl_hf_to_auto.max())

    # Count outliers at different thresholds
    total = diff.numel()
    outliers_0_1 = (diff > 0.1).sum().item()
    outliers_1_0 = (diff > 1.0).sum().item()

    results = []
    any_match = False
    for rtol, atol, name in tolerances:
        match = torch.allclose(automodel_logits, hf_logits, rtol=rtol, atol=atol)
        results.append((rtol, atol, name, bool(match)))
        if match:
            any_match = True
            break

    # Print comparison results
    print("\n" + "=" * 70)
    print("LOGPROBS COMPARISON")
    print("=" * 70)
    print(f"Max abs diff:    {max_abs:.6f}")
    print(f"Mean abs diff:   {mean_abs:.6f}")
    print(f"Median abs diff: {median_abs:.6f}")
    print(f"Max rel diff:    {max_rel:.6f}")
    print(f"Mean rel diff:   {mean_rel:.6f}")

    # Print outlier info if significant
    if outliers_1_0 > 0:
        print(f"Outliers >1.0:   {outliers_1_0:6d} ({100.0 * outliers_1_0 / total:5.2f}%)")
    elif outliers_0_1 > 0:
        print(f"Outliers >0.1:   {outliers_0_1:6d} ({100.0 * outliers_0_1 / total:5.2f}%)")

    # Print token-wise KL divergence statistics
    print("\nToken-wise KL Divergence:")
    print(
        f"  KL(automodel || hf): mean={kl_auto_to_hf_mean:.6f}, median={kl_auto_to_hf_median:.6f}, max={kl_auto_to_hf_max:.6f}"
    )
    print(
        f"  KL(hf || automodel): mean={kl_hf_to_auto_mean:.6f}, median={kl_hf_to_auto_median:.6f}, max={kl_hf_to_auto_max:.6f}"
    )

    print("\nTolerance tests:")
    for rtol, atol, name, match in results:
        print(f"  {'✅' if match else '❌'} {name} (rtol={rtol}, atol={atol})")

    if not any_match:
        flat = (automodel_logits - hf_logits).flatten()
        mean_offset = float(torch.mean(flat))
        std_offset = float(torch.std(flat))
        print(f"\nSystematic offset: mean={mean_offset:.6f}, std={std_offset:.6f}")


def compare_activation_dicts(
    auto_acts: Dict[str, torch.Tensor],
    hf_acts: Dict[str, torch.Tensor],
    processing_functions: Optional[Dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
    model_cls: str = None,
) -> None:
    if processing_functions is None:
        processing_functions = {}

    # Apply processing functions to matching keys
    import re

    for pattern, func in processing_functions.items():
        # Compile regex pattern
        try:
            regex = re.compile(pattern)
        except re.error:
            logger.warning(f"Invalid regex pattern: {pattern}")
            continue

        for key in list(auto_acts.keys()):
            if regex.search(key):
                try:
                    auto_acts[key] = func(auto_acts[key])
                except Exception as e:
                    logger.warning(f"Failed to process automodel activation {key}: {e}")

        for key in list(hf_acts.keys()):
            if regex.search(key):
                try:
                    hf_acts[key] = func(hf_acts[key])
                except Exception as e:
                    logger.warning(f"Failed to process HF activation {key}: {e}")

    keys = sorted(set(auto_acts.keys()) & set(hf_acts.keys()))
    if not keys:
        print("No overlapping activation keys to compare.")
        return

    # Group keys by layer number
    import re

    layer_pattern = re.compile(r"\.layers\.(\d+)")

    # Separate into layer groups and non-layer components
    layer_groups: Dict[int, list] = {}
    non_layer_keys = []

    for key in keys:
        match = layer_pattern.search(key)
        if match:
            layer_num = int(match.group(1))
            if layer_num not in layer_groups:
                layer_groups[layer_num] = []
            layer_groups[layer_num].append(key)
        else:
            non_layer_keys.append(key)

    print("\n" + "=" * 90)
    print("ACTIVATION COMPARISON")
    print("=" * 90)

    # Print non-layer components first (embed_tokens, lm_head, norm, etc.)
    if non_layer_keys:
        print("\n[MODEL COMPONENTS]")
        print("-" * 90)
        for key in sorted(non_layer_keys):
            _process_and_compare_activation(key, auto_acts[key], hf_acts[key], model_cls)

    # Print layer-by-layer comparisons
    for layer_num in sorted(layer_groups.keys()):
        print(f"\n[LAYER {layer_num}]")
        print("-" * 90)

        # Sort keys within each layer for better organization
        # First print the layer-level activation, then its subcomponents
        layer_keys = layer_groups[layer_num]

        # Separate layer-level key from subcomponents
        layer_root = None
        subcomponents = []
        for key in layer_keys:
            if key == f"model.layers.{layer_num}":
                layer_root = key
            else:
                subcomponents.append(key)

        # Print layer root first
        if layer_root:
            _process_and_compare_activation(
                layer_root, auto_acts[layer_root], hf_acts[layer_root], model_cls, indent="  "
            )

        # Sort subcomponents for hierarchical display
        for key in sorted(subcomponents):
            # Add indentation for subcomponents
            _process_and_compare_activation(key, auto_acts[key], hf_acts[key], model_cls, indent="  ")


def _compare_single_activation(key: str, a: torch.Tensor, b: torch.Tensor, indent: str = "  ") -> None:
    a = a.squeeze(0)
    b = b.squeeze(0)

    if a.shape != b.shape:
        print(f"{indent}{key}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        return

    # Use full FQN
    display_name = key

    # Color codes for highlighting issues
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    # Standard comparison for float tensors
    diff = (a - b).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    # Count outliers at different thresholds
    total = a.numel()
    outliers_0_1 = (diff > 0.1).sum().item()
    outliers_1_0 = (diff > 1.0).sum().item()

    # Determine color based on absolute error thresholds
    color = ""
    if max_abs > 1.0 and mean_abs > 0.05:
        color = RED
    elif max_abs > 0.1 and mean_abs > 0.01:
        color = YELLOW

    # Build output with outlier info if significant
    outlier_str = ""
    if outliers_1_0 > 0:
        outlier_str = f" | outliers>1.0: {outliers_1_0:6d} ({100.0 * outliers_1_0 / total:5.2f}%)"
    elif outliers_0_1 > 0:
        outlier_str = f" | outliers>0.1: {outliers_0_1:6d} ({100.0 * outliers_0_1 / total:5.2f}%)"

    print(f"{color}{indent}{display_name:60s} | max_abs={max_abs:8.6f} mean_abs={mean_abs:8.6f}{outlier_str}{RESET}")


def _compare_gate_activation(
    key: str, auto_tuple: tuple, hf_tuple: tuple, gate_process_fn: Callable[[tuple, tuple], tuple], indent: str = "  "
) -> None:
    if len(auto_tuple) != len(hf_tuple):
        print(f"{indent}{key}: tuple length mismatch {len(auto_tuple)} vs {len(hf_tuple)}")
        return

    hf_tuple, auto_tuple = gate_process_fn(hf_tuple, auto_tuple)

    # Color codes for highlighting issues
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    # Compare indices (element 0)
    a_indices = auto_tuple[0].squeeze(0)
    b_indices = hf_tuple[0].squeeze(0)

    if a_indices.shape != b_indices.shape:
        print(f"{indent}{key}[0]: shape mismatch {tuple(a_indices.shape)} vs {tuple(b_indices.shape)}")
    else:
        mismatches = (a_indices != b_indices).sum().item()
        total = a_indices.numel()
        mismatch_pct = 100.0 * mismatches / total if total > 0 else 0.0

        # Color based on mismatch percentage
        color = ""
        if mismatch_pct > 10.0:
            color = RED
        elif mismatch_pct > 1.0:
            color = YELLOW

        display_name = f"{key}[0]"
        print(
            f"{color}{indent}{display_name:60s} | mismatches={mismatches:6d}/{total:6d} ({mismatch_pct:6.2f}%){RESET}"
        )

    # Compare weights (element 1) using standard float comparison
    _compare_single_activation(f"{key}[1]", auto_tuple[1].float(), hf_tuple[1].float(), indent=indent)


def _process_and_compare_activation(
    key: str, auto_act: torch.Tensor, hf_act: torch.Tensor, model_cls: str, indent: str = "  "
) -> None:
    # Handle tuple activations (e.g., from gate/router modules)
    if isinstance(auto_act, tuple) and isinstance(hf_act, tuple):
        _compare_gate_activation(key, auto_act, hf_act, GATE_PROCESS_FUNCTIONS[model_cls], indent=indent)
    else:
        # Handle single tensor activations
        if isinstance(auto_act, tuple):
            auto_act = auto_act[0]
        if isinstance(hf_act, tuple):
            hf_act = hf_act[0]

        _compare_single_activation(key, auto_act.float(), hf_act.float(), indent=indent)


def truncate_hf_layers(hf_model: nn.Module, max_layers: int) -> bool:
    import gc

    try_paths = [
        ("model", "layers"),
        ("transformer", "layers"),
        ("model", "decoder", "layers"),
    ]
    for path in try_paths:
        parent = hf_model
        ok = True
        for attr in path:
            if hasattr(parent, attr):
                parent = getattr(parent, attr)
            else:
                ok = False
                break
        if not ok:
            continue
        layers = parent
        if isinstance(layers, nn.ModuleList) and len(layers) > max_layers:
            kept = nn.ModuleList([layers[i] for i in range(max_layers)])
            # Replace in the actual parent container
            # Walk again to set attribute
            container = hf_model
            for attr in path[:-1]:
                container = getattr(container, attr)
            setattr(container, path[-1], kept)
            # Update config if present
            if hasattr(hf_model, "config") and hasattr(hf_model.config, "num_hidden_layers"):
                hf_model.config.num_hidden_layers = max_layers
            del layers
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
    return False


# ============================================================================
# Text Generation Utilities
# ============================================================================


@torch.no_grad()
def generate_text(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    # Ensure batch dimension
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    generated_tokens = input_ids.clone()

    for _ in range(max_new_tokens):
        out = model(generated_tokens)
        logits = out.logits if hasattr(out, "logits") else out
        next_token_logits = logits[:, -1, :]
        # Greedy: pick argmax token deterministically
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).any():
            break

    return generated_tokens


class GenerationAndComparisonRecipeForCausalLM(BenchmarkingRecipeForNextTokenPrediction):
    """Recipe for text generation with optional HuggingFace comparison.

    This recipe does text generation with Automodel and optionally compares the results with HuggingFace.
    """

    def __init__(self, cfg):
        # Store generation-specific parameters
        gen_cfg = cfg.generation
        self._prompt = gen_cfg.get("prompt", "Once upon a time")
        self._max_new_tokens = gen_cfg.get("max_new_tokens", 100)
        self._seed = gen_cfg.get("seed", 42)

        # HuggingFace comparison settings
        self._compare_hf = gen_cfg.get("compare_hf", False)
        self._hf_model_path = gen_cfg.get("hf_model_path", None)

        # Activation debugging settings
        self._debug_activations = gen_cfg.get("debug_activations", False)
        self._activation_processing = gen_cfg.get("activation_processing_functions", {})

        # Layer limiting - override model config if max_layers is set
        max_layers = gen_cfg.get("max_layers", None)
        self._max_layers = max_layers  # Store for later use in HF model truncation
        if max_layers is not None and max_layers > 0:
            if not hasattr(cfg.model, "config"):
                # If no config exists, create one
                from transformers import AutoConfig

                # Handle both config.pretrained_model_name_or_path and model.pretrained_model_name_or_path
                if hasattr(cfg.model, "pretrained_model_name_or_path"):
                    model_name = cfg.model.pretrained_model_name_or_path
                else:
                    raise ValueError("Could not find pretrained_model_name_or_path in model config")

                cfg.model.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

            # Override num_hidden_layers
            if hasattr(cfg.model.config, "_target_"):
                # Config is a hydra instantiation, need to set it as a parameter
                cfg.model.config.num_hidden_layers = max_layers
            else:
                # Direct config object
                cfg.model.config.num_hidden_layers = max_layers

            # Log only on rank 0
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                logger.info(f"Limiting model to first {max_layers} layers for generation")

        # Call parent init
        super().__init__(cfg)

    def setup(self):
        # Call parent setup to initialize model, distributed env, etc.
        # Note: This will try to create optimizer and dataloader, which we'll clear later
        # The parent already wraps this in a timer, so we don't need to wrap it again
        super().setup()

        # Set models to eval mode (override training mode from parent)
        for mp in self.model_parts:
            mp.eval()

        # Clear unnecessary state - we don't need optimizer or dataloaders for generation
        self.optimizer = None
        self.dataloader = None
        self.val_dataloader = None

        # Load tokenizer (if not already loaded by parent)
        # Handle both config.pretrained_model_name_or_path and model.pretrained_model_name_or_path
        if hasattr(self.cfg.model, "pretrained_model_name_or_path"):
            self._model_name = self.cfg.model.pretrained_model_name_or_path
        elif hasattr(self.cfg.model, "config") and hasattr(self.cfg.model.config, "pretrained_model_name_or_path"):
            self._model_name = self.cfg.model.config.pretrained_model_name_or_path
        else:
            raise ValueError("Could not find pretrained_model_name_or_path in model config")

        # Only load tokenizer if it wasn't already loaded by parent
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        for mp in self.model_parts:
            self.checkpointer.load_base_model(
                mp,
                self.dist_env.device,
                self.cfg.model.get("cache_dir", TRANSFORMERS_CACHE),
                self._model_name,
                None,
                load_base_model=True,
            )

    def run_generation_with_comparison(self):
        rank = self.dist_env.rank
        device = self.dist_env.device

        # Determine if we have the first stage (for pipeline parallelism)
        # If pp is disabled, we assume we have the full model
        has_first_stage = not self.pp_enabled or (hasattr(self, "has_first_stage") and self.has_first_stage)

        # Encode prompt
        input_ids = self._tokenizer.encode(self._prompt, add_special_tokens=False, return_tensors="pt")
        if self.dist_env.is_main:
            logger.info(f"Prompt: {self._prompt}")
            logger.info(f"Input tokens: {input_ids.shape[1]}")

        # ====================================================================
        # Generate with Automodel
        # ====================================================================
        automodel_logits = None
        automodel_generated_ids = None
        auto_recorder: Optional[ActivationRecorder] = None

        if has_first_stage:
            if self.dist_env.is_main:
                if self._debug_activations:
                    logger.info("Recording activations with Automodel...")
                else:
                    logger.info("Generating text with Automodel...")

            gen_model = self.model_parts[0]

            # Register activation hooks if requested
            if self._debug_activations:
                auto_recorder = ActivationRecorder()
                auto_recorder.register(gen_model)

            # Get logits for comparison
            with torch.no_grad():
                automodel_logits = gen_model(input_ids.to(device)).detach().cpu()

            # Generate text (skip if debugging activations to save time)
            if not self._debug_activations:
                automodel_generated_ids = generate_text(
                    gen_model,
                    input_ids.to(device),
                    max_new_tokens=self._max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                generated_text = self.tokenizer.decode(automodel_generated_ids[0].tolist(), skip_special_tokens=True)
                new_tokens = automodel_generated_ids.shape[1] - input_ids.shape[1]

                if self.dist_env.is_main:
                    print("\n" + "=" * 60)
                    print(f"RANK {rank} | AUTOMODEL GENERATED TEXT")
                    print("=" * 60)
                    print(generated_text)
                    print("=" * 60)
                    print(f"\nGenerated {new_tokens} new tokens")
                    print(f"Total sequence length: {automodel_generated_ids.shape[1]} tokens")

            # Move model to CPU to free GPU memory
            for mp in self.model_parts:
                mp.to("cpu")
            torch.cuda.empty_cache()

        # ====================================================================
        # Compare with HuggingFace
        # ====================================================================
        if self._compare_hf and self.dist_env.is_main:
            hf_model_path = self._hf_model_path or self._model_name
            logger.info(f"\nLoading HuggingFace model from {hf_model_path}...")

            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            # Default to bfloat16 if no device config
            dtype = torch.bfloat16
            if hasattr(self.cfg, "device") and hasattr(self.cfg.device, "dtype"):
                dtype = dtype_map.get(self.cfg.device.dtype, torch.bfloat16)

            hf_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="cpu",
            )

            # Optionally truncate HF to first N layers to save memory
            if self._max_layers is not None:
                did_truncate = truncate_hf_layers(hf_model, self._max_layers)
                if did_truncate:
                    logger.info(f"Truncated HF model to first {self._max_layers} layers for activation comparison.")

            # Move to GPU and set to eval
            hf_model = hf_model.to(hf_device)
            with hf_device:
                if hasattr(hf_model.model, "rotary_emb"):
                    # Rope frequencies need to be reinitialized on the GPU, as CPU initialization has high numerical differences with GPU initialization.
                    hf_model.model.rotary_emb.inv_freq, hf_model.model.rotary_emb.attention_scaling = (
                        hf_model.model.rotary_emb.rope_init_fn(hf_model.model.rotary_emb.config, hf_device)
                    )
                else:
                    for layer in hf_model.model.layers:
                        if hasattr(layer.self_attn, "rotary_emb") and hasattr(
                            layer.self_attn.rotary_emb, "rope_init_fn"
                        ):
                            layer.self_attn.rotary_emb.inv_freq, layer.self_attn.rotary_emb.attention_scaling = (
                                layer.self_attn.rotary_emb.rope_init_fn(layer.self_attn.rotary_emb.config, hf_device)
                            )
                        elif hasattr(layer.self_attn, "_init_rope"):
                            layer.self_attn._init_rope()

            hf_model.eval()

            # Register activation hooks if requested
            hf_recorder: Optional[ActivationRecorder] = None
            if self._debug_activations:
                hf_recorder = ActivationRecorder()
                hf_recorder.register(hf_model)

            # Get HF logits
            with torch.no_grad():
                hf_out = hf_model(input_ids.to(hf_device), use_cache=False)
                hf_logits = hf_out.logits if hasattr(hf_out, "logits") else hf_out

            # Compare logits
            if automodel_logits is not None:
                compare_and_print_logits(automodel_logits.to(hf_device), hf_logits)

            # Generate with HF (skip if debugging activations)
            if not self._debug_activations:
                logger.info("\nGenerating text with HuggingFace model...")
                hf_generated_ids = generate_text(
                    hf_model,
                    input_ids.to(hf_device),
                    max_new_tokens=self._max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                hf_generated_text = self.tokenizer.decode(hf_generated_ids[0].tolist(), skip_special_tokens=True)
                print("\n" + "=" * 60)
                print("HUGGINGFACE GENERATED TEXT")
                print("=" * 60)
                print(hf_generated_text)
                print("=" * 60)

            # Compare activations
            if self._debug_activations and auto_recorder is not None and hf_recorder is not None:
                compare_activation_dicts(
                    auto_recorder.activations,
                    hf_recorder.activations,
                    processing_functions=self._activation_processing,
                    model_cls=hf_model.__class__.__name__,
                )
                auto_recorder.remove()
                hf_recorder.remove()

            # Cleanup HF model
            del hf_model
            gc.collect()
            torch.cuda.empty_cache()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def main(config_path=None):
    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

    if config_path is None:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True, help="Path to config file")
        args, _ = parser.parse_known_args()
        config_path = args.config

    cfg = parse_args_and_load_config(config_path)
    recipe = GenerationAndComparisonRecipeForCausalLM(cfg)
    recipe.setup()
    recipe.run_generation_with_comparison()


if __name__ == "__main__":
    main()
