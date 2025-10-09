from typing import Dict, List

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch, MixedPrecision, ShardingStrategy

from .dist_utils import cast_model_to_dtype, print0
from .lora_utils import (
    collect_wan_lora_parameters,
    wan_install_and_materialize_lora,
)


def setup_hybrid_for_pipe(pipe, device, bf16, local_rank: int, lora_rank: int, lora_alpha: int):
    """
    Returns:
      model_map: {
        name: {
          "fsdp": fsdp_wrapped_module,
          "base_transformer": original_module,
          "lora_params": List[nn.Parameter],
        }
      }
      transformer_names: List[str]
    """
    fsdp_mixed_precision = MixedPrecision(param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16)

    transformer_names: List[str] = []
    for name in ["transformer", "transformer_2"]:
        if hasattr(pipe, name) and getattr(pipe, name) is not None:
            transformer_names.append(name)
    if not transformer_names:
        raise RuntimeError("No transformers found in pipeline")

    print0(f"[INFO] Found transformers: {transformer_names}")

    model_map: Dict[str, Dict] = {}

    for name in transformer_names:
        base_transformer = getattr(pipe, name)

        # Cast & move base
        cast_model_to_dtype(base_transformer, bf16)
        base_transformer.to(device)

        # Materialize real LoRA parameters inside WAN transformer
        wan_install_and_materialize_lora(base_transformer, rank=lora_rank, alpha=lora_alpha)

        # Collect only LoRA params (trainable)
        lora_params = collect_wan_lora_parameters(base_transformer)
        if len(lora_params) == 0:
            raise RuntimeError(f"No LoRA parameters found for {name}")

        # Wrap base in FSDP
        fsdp_wrapped = FSDP(
            base_transformer,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=fsdp_mixed_precision,
            auto_wrap_policy=None,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=local_rank,
            sync_module_states=True,
            use_orig_params=False,
        )
        fsdp_wrapped.train()

        model_map[name] = {
            "fsdp": fsdp_wrapped,
            "base_transformer": base_transformer,
            "lora_params": lora_params,
        }
        setattr(pipe, name, fsdp_wrapped)

    print0("[INFO] Hybrid parallelism setup complete")
    return model_map, transformer_names
