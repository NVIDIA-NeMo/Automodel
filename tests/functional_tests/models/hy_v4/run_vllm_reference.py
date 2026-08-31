# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture HY V4 hidden states and logits from the pinned vLLM reference.

This utility is intentionally separate from the AutoModel runtime. It requires
an external vLLM environment whose HY V4 Python sources match the hashes from
vLLM commit ``b2f685834a6456197e7033966fdef52a23f1abcd``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from safetensors.torch import save_file

VLLM_REFERENCE_COMMIT = "b2f685834a6456197e7033966fdef52a23f1abcd"
VLLM_SOURCE_HASHES = {
    "vllm.models.hy_v4.nvidia.attention": "a8fede3a28ca485acdd8ff405018e8a5db466c9f8049c43a0d777284366e2cdd",
    "vllm.models.hy_v4.nvidia.flashmla_sparse": "2135038001d03c02cbbe4c97d6ac0dfa520d59b18cd2e59e0818747060a041ae",
    "vllm.models.hy_v4.nvidia.hc": "d20ff026825125df44bb3f14760f5cf77ae8c2a7a17a81c29c0c03c498232454",
    "vllm.models.hy_v4.nvidia.model": "f48b8e9c683a8be05e23e1ad10a5ca2e31ffed05c605db2d6251ef786bdf9f36",
    "vllm.models.hy_v4.nvidia.moe": "8d55295e8d0b49efea2ad7f52a7d7e3f3154260eb28e04463fac935cfdfd7d70",
    "vllm.models.hy_v4.nvidia.mtp": "60a98496f3fdcc5aaaedb758099c91997550299f62951b774e09ca48f71e0605",
    "vllm.transformers_utils.configs.hy_v4": "73c98c51d67a7466ee942560fbd1309e1f25b0a7d7ca858ce1df254a6c655c56",
}
DEFAULT_TOKEN_IDS = [1, 42, 314, 1592, 2718, 4096, 8191, 120000]
HY_V4_VOCAB_SIZE = 120832
HY_V4_BOS_TOKEN_ID = 120000


def _module_source_path(module: ModuleType) -> Path:
    source_path = module.__file__
    if source_path is None:
        raise RuntimeError(f"Imported module has no source path: {module.__name__}")
    return Path(source_path)


def _verify_vllm_source() -> dict[str, str]:
    """Verify that imported HY V4 sources are the pinned vLLM implementation."""
    resolved_sources: dict[str, str] = {}
    for module_name, expected_hash in VLLM_SOURCE_HASHES.items():
        module = importlib.import_module(module_name)
        source_path = _module_source_path(module)
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"vLLM source mismatch for {module_name}: expected {expected_hash}, got {actual_hash} at {source_path}."
            )
        resolved_sources[module_name] = str(source_path.resolve())
    return resolved_sources


def _validate_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Validate the proxy checkpoint provenance against the vLLM oracle."""
    provenance_path = checkpoint / "reference_provenance.json"
    with provenance_path.open(encoding="utf-8") as stream:
        provenance = json.load(stream)
    actual_commit = provenance.get("vllm_reference_commit")
    if actual_commit != VLLM_REFERENCE_COMMIT:
        raise RuntimeError(
            f"Checkpoint reference mismatch: expected vLLM {VLLM_REFERENCE_COMMIT}, got {actual_commit}."
        )
    return provenance


def _capture_tensor(output: torch.Tensor, *, name: str) -> torch.Tensor:
    """Copy a captured vLLM tensor to owned CPU storage.

    Args:
        output: Tensor of arbitrary shape produced by the reference model.
        name: Human-readable tensor name used in validation errors.

    Returns:
        Tensor with the same shape and dtype on contiguous CPU storage.
    """
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected {name} to be a Tensor, got {type(output).__name__}.")
    return output.detach().contiguous().cpu().clone()


def run_reference(
    checkpoint: Path,
    output_path: Path,
    token_ids: list[int],
    *,
    enable_hpc_ops: bool,
    capture_all_token_logits: bool = False,
    logits_chunk_size: int = 128,
    gpu_memory_utilization: float = 0.5,
) -> None:
    """Run one prompt through vLLM and persist projection inputs and logits.

    Args:
        checkpoint: Local HY V4 proxy checkpoint containing unmodified public weights.
        output_path: Destination safetensors path for captured tensors.
        token_ids: Token IDs represented as a Python list of shape [sequence].
        enable_hpc_ops: Whether vLLM may use its fused HPC iHC operators.
        capture_all_token_logits: Project every prompt position over the full vocabulary.
        logits_chunk_size: Number of token rows per full-vocabulary projection.
        gpu_memory_utilization: Fraction of device memory vLLM may reserve.
    """
    if not token_ids:
        raise ValueError("token_ids must contain at least one token.")
    if logits_chunk_size < 1:
        raise ValueError(f"logits_chunk_size must be positive, got {logits_chunk_size}.")
    metadata_path = output_path.with_suffix(".json")
    existing_outputs = [path for path in (output_path, metadata_path) if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite reference artifacts: {existing_outputs}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_provenance = _validate_checkpoint(checkpoint)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ENABLE_HPC_OPS"] = "1" if enable_hpc_ops else "0"

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    resolved_sources = _verify_vllm_source()
    captures: dict[str, list[torch.Tensor]] = {
        "full_hidden_states": [],
        "full_logits": [],
        "sample_hidden_states": [],
        "sample_logits": [],
    }
    llm: LLM | None = None
    hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    request_outputs: list[Any] = []
    full_projection_active = False
    try:
        llm = LLM(
            model=str(checkpoint),
            tokenizer=str(checkpoint),
            skip_tokenizer_init=True,
            trust_remote_code=False,
            dtype="bfloat16",
            attention_config={
                "backend": AttentionBackendEnum.FLASHMLA_SPARSE,
                "sparse_mla_force_mqa": True,
            },
            enforce_eager=True,
            max_model_len=max(16, len(token_ids) + 1),
            max_num_seqs=1,
            max_num_batched_tokens=max(16, len(token_ids) + 1),
            gpu_memory_utilization=gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_log_stats=True,
            load_format="safetensors",
            seed=1234,
        )
        model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
        model = model_runner.model

        def capture_full_hidden(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            """Capture backbone output of shape [tokens, hidden].

            Args:
                _module: HY V4 backbone module that produced the tensor.
                _inputs: Positional model inputs containing tensors in flattened token layout.
                output: Tensor of shape [tokens, hidden] after final normalization.
            """
            nonlocal full_projection_active
            captures["full_hidden_states"].append(_capture_tensor(output, name="full hidden states"))
            if capture_all_token_logits:
                full_logits: list[torch.Tensor] = []
                full_projection_active = True
                try:
                    for hidden_chunk in output.split(logits_chunk_size, dim=0):
                        logits_chunk = model.compute_logits(hidden_chunk)
                        if logits_chunk is None:
                            raise RuntimeError("vLLM returned no logits for a full-prompt projection chunk.")
                        full_logits.append(_capture_tensor(logits_chunk, name="full logits chunk"))
                finally:
                    full_projection_active = False
                captures["full_logits"].append(torch.cat(full_logits, dim=0))

        def capture_logits(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            """Capture the sampled hidden rows and their vocabulary logits.

            Args:
                _module: vLLM logits processor module.
                inputs: Positional arguments whose second item is a Tensor of shape [sampled_tokens, hidden].
                output: Tensor of shape [sampled_tokens, vocab].
            """
            if full_projection_active:
                return
            captures["sample_hidden_states"].append(_capture_tensor(inputs[1], name="sample hidden states"))
            captures["sample_logits"].append(_capture_tensor(output, name="sample logits"))

        hook_handles.append(model.model.register_forward_hook(capture_full_hidden))
        hook_handles.append(model.logits_processor.register_forward_hook(capture_logits))
        request_outputs = llm.generate(
            [{"prompt_token_ids": token_ids}],
            SamplingParams(max_tokens=1, temperature=0.0),
            use_tqdm=False,
        )
    finally:
        for handle in hook_handles:
            handle.remove()
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()

    capture_counts = {name: len(values) for name, values in captures.items()}
    expected_counts = {
        "full_hidden_states": 1,
        "full_logits": int(capture_all_token_logits),
        "sample_hidden_states": 1,
        "sample_logits": 1,
    }
    if capture_counts != expected_counts:
        raise RuntimeError(f"Expected exactly one vLLM forward, got capture counts {capture_counts}.")
    if len(request_outputs) != 1 or len(request_outputs[0].outputs) != 1:
        raise RuntimeError("vLLM did not return exactly one completion for the parity prompt.")

    tensors = {
        "input_ids": torch.tensor(token_ids, dtype=torch.int64),
        "full_hidden_states": captures["full_hidden_states"][0],
        "sample_hidden_states": captures["sample_hidden_states"][0],
        "sample_logits": captures["sample_logits"][0],
        "logits": (captures["full_logits"][0] if capture_all_token_logits else captures["sample_logits"][0]),
    }
    save_file(tensors, output_path, metadata={"format": "pt"})
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_provenance": checkpoint_provenance,
        "capture_counts": capture_counts,
        "cuda_version": torch.version.cuda,
        "generated_token_ids": list(request_outputs[0].outputs[0].token_ids),
        "hpc_ops_enabled": enable_hpc_ops,
        "capture_all_token_logits": capture_all_token_logits,
        "gpu_memory_utilization": gpu_memory_utilization,
        "logits_chunk_size": logits_chunk_size,
        "tensor_dtypes": {name: str(tensor.dtype) for name, tensor in tensors.items()},
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "torch_version": torch.__version__,
        "vllm_reference_commit": VLLM_REFERENCE_COMMIT,
        "vllm_source_paths": resolved_sources,
        "vllm_version": vllm.__version__,
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    token_group = parser.add_mutually_exclusive_group()
    token_group.add_argument("--token-ids", type=int, nargs="+")
    token_group.add_argument(
        "--sequence-length",
        type=int,
        help="Build a deterministic random-token prompt of this length (position zero is BOS).",
    )
    parser.add_argument("--token-seed", type=int, default=1234)
    parser.add_argument("--enable-hpc-ops", action="store_true")
    parser.add_argument("--capture-all-token-logits", action="store_true")
    parser.add_argument("--logits-chunk-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()
    if args.sequence_length is not None:
        if args.sequence_length < 1:
            parser.error("--sequence-length must be positive")
        generator = torch.Generator(device="cpu").manual_seed(args.token_seed)
        token_ids = torch.randint(
            0,
            HY_V4_VOCAB_SIZE,
            (args.sequence_length,),
            generator=generator,
            dtype=torch.int64,
        ).tolist()
        token_ids[0] = HY_V4_BOS_TOKEN_ID
    else:
        token_ids = args.token_ids or DEFAULT_TOKEN_IDS
    run_reference(
        args.checkpoint,
        args.output,
        token_ids,
        enable_hpc_ops=args.enable_hpc_ops,
        capture_all_token_logits=args.capture_all_token_logits,
        logits_chunk_size=args.logits_chunk_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
