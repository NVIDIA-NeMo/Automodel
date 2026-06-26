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

"""vLLM forward for the EAGLE-3 target (server-side, GPU only).

This module owns every vLLM-internal touch point so the rest of the speculative
stack stays vLLM-agnostic and importable without vLLM. It is imported lazily
(only from :meth:`VLLMEagle3TargetModel.from_pretrained`) and implements the
engine-agnostic
:class:`~nemo_automodel.components.speculative.eagle.target_runner.TargetRunner`
surface, exactly like ``sglang_runner.SGLangTargetRunner``.

Mechanism (vLLM's native ``extract_hidden_states`` speculative method, the
supported way to pull EAGLE-3 supervision out of vLLM without driving v1 worker
internals):

1. Build an offline ``LLM`` with
   ``speculative_config={"method": "extract_hidden_states", ...}`` and the three
   EAGLE-3 capture layers (plus ``num_hidden_layers`` so the final pre-norm
   hidden is captured too) in
   ``draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids``. Chunked
   prefill is disabled so every prompt is captured in one prefill.
2. ``generate(max_tokens=1)`` over the batch; vLLM writes the captured hidden
   states to disk through ``ExampleHiddenStatesConnector`` (a KV connector), and
   each request output carries the path under ``kv_transfer_params``.
3. Read the per-prompt ``[seq, num_capture_layers, hidden]`` tensor back, split
   the three EAGLE-3 layers (concatenated into ``[seq, 3 * hidden]``) from the
   final-layer hidden, and rebuild full-vocab logits in-process by applying the
   target's final RMSNorm + LM head (loaded once from the model's safetensors).

vLLM's ``extract_hidden_states`` API is version-coupled: the calls here track
``vllm==0.23.0``. This forward path requires a GPU and vLLM, so it is validated
on the training server, not in CPU unit tests; the CPU tests exercise the
shared contract layer against a fake runner instead.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Optional, Sequence

import torch

logger = logging.getLogger(__name__)

_VLLM_DTYPE_STRINGS = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}


def vllm_dtype_str(dtype: Optional[torch.dtype]) -> str:
    """Map a torch dtype to the string form vLLM's ``dtype`` argument expects.

    ``None`` means "let vLLM pick" (``"auto"``).
    """
    if dtype is None:
        return "auto"
    if dtype not in _VLLM_DTYPE_STRINGS:
        raise ValueError(f"Unsupported vLLM target dtype {dtype}; expected one of {list(_VLLM_DTYPE_STRINGS)}.")
    return _VLLM_DTYPE_STRINGS[dtype]


def _load_target_head_weights(
    model_path: str, device: torch.device
):  # pragma: no cover - reads model safetensors on the GPU server
    """Load ``(embed, final_norm, lm_head)`` weights straight from the safetensors.

    The real weights live inside the vLLM engine (a separate process), so the
    final-norm + LM-head used to rebuild logits, and the input embeddings the
    draft copies, are read directly off disk here instead. ``lm_head`` falls back
    to the input embeddings for tied-embedding models (e.g. Qwen3).
    """
    from safetensors import safe_open

    want = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    index = os.path.join(model_path, "model.safetensors.index.json")
    key_to_file: dict[str, str] = {}
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        for key in want:
            if key in weight_map:
                key_to_file[key] = os.path.join(model_path, weight_map[key])
    else:
        for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            with safe_open(shard, framework="pt") as sf:
                for key in sf.keys():
                    if key in want:
                        key_to_file[key] = shard

    tensors: dict[str, torch.Tensor] = {}
    by_file: dict[str, list[str]] = {}
    for key, shard in key_to_file.items():
        by_file.setdefault(shard, []).append(key)
    for shard, keys in by_file.items():
        with safe_open(shard, framework="pt", device="cpu") as sf:
            for key in keys:
                tensors[key] = sf.get_tensor(key)

    embed = tensors.get("model.embed_tokens.weight")
    norm = tensors.get("model.norm.weight")
    if embed is None or norm is None:
        raise RuntimeError(
            f"Could not load model.embed_tokens.weight / model.norm.weight from {model_path}; "
            f"found keys {sorted(tensors)}."
        )
    lm_head = tensors.get("lm_head.weight", embed)  # tied embeddings -> reuse the input embedding
    return embed.to(device), norm.to(device), lm_head.to(device)


class _VLLMModelShim:
    """Lightweight stand-in exposing ``.config`` + ``.parameters()``.

    The target's real parameters live in the vLLM engine process, so this only
    carries the HF config (for ``num_hidden_layers`` / ``hidden_size`` /
    ``vocab_size``) and a single device-marker tensor so the remote server can
    still infer the target's device via ``next(model.parameters()).device``.
    """

    def __init__(self, hf_config, device: torch.device):
        self.config = hf_config
        self._marker = torch.empty(0, device=device)

    def parameters(self):
        yield self._marker


class VLLMTargetRunner:
    """Offline vLLM engine that returns EAGLE-3 supervision tensors.

    Built via :meth:`build`; consumed through the engine-agnostic
    :class:`~nemo_automodel.components.speculative.eagle.target_runner.TargetRunner`
    surface (``model`` / ``set_aux_layers`` / ``forward_eagle3`` /
    ``input_embedding_weight``). The vLLM ``LLM`` is constructed lazily in
    :meth:`set_aux_layers`, because the capture layers must be known up front.
    """

    def __init__(
        self,
        model_path: str,
        *,
        dtype: Optional[torch.dtype] = None,
        tp_size: int = 1,
        trust_remote_code: bool = False,
        gpu_memory_utilization: float = 0.5,
        shared_storage_path: Optional[str] = None,
        vllm_kwargs: Optional[dict] = None,
    ):
        from transformers import AutoConfig

        self._model_path = model_path
        self._dtype = dtype
        self._tp_size = tp_size
        self._trust_remote_code = trust_remote_code
        self._gpu_memory_utilization = gpu_memory_utilization
        self._shared_storage_path = shared_storage_path or os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "vllm_eagle3_hidden_states"
        )
        self._vllm_kwargs = dict(vllm_kwargs or {})

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self._hf_config = hf_config
        self._num_layers = int(hf_config.num_hidden_layers)
        self._hidden = int(hf_config.hidden_size)
        self._rms_eps = float(getattr(hf_config, "rms_norm_eps", 1e-6))
        self.model = _VLLMModelShim(hf_config, torch.device("cuda", torch.cuda.current_device()))

        self._llm = None
        self._aux_layer_ids: Optional[list[int]] = None
        self._embed_w: Optional[torch.Tensor] = None
        self._norm_w: Optional[torch.Tensor] = None
        self._lm_head_w: Optional[torch.Tensor] = None

    @classmethod
    def build(  # pragma: no cover - requires GPU + vLLM
        cls,
        model_path: str,
        *,
        dtype: Optional[torch.dtype] = None,
        tp_size: int = 1,
        trust_remote_code: bool = False,
        **vllm_kwargs,
    ) -> "VLLMTargetRunner":
        """Construct the runner for a standalone target server.

        ``vllm_kwargs`` are forwarded to ``vllm.LLM`` (e.g. ``gpu_memory_utilization``,
        ``max_model_len``, ``quantization``). GPU/vLLM-only.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("VLLMTargetRunner requires CUDA; run it on a GPU server, not the editing host.")
        gpu_memory_utilization = vllm_kwargs.pop("gpu_memory_utilization", 0.5)
        shared_storage_path = vllm_kwargs.pop("shared_storage_path", None)
        return cls(
            model_path,
            dtype=dtype,
            tp_size=tp_size,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            shared_storage_path=shared_storage_path,
            vllm_kwargs=vllm_kwargs,
        )

    def set_aux_layers(self, aux_layer_ids: Sequence[int]) -> None:  # pragma: no cover - requires GPU + vLLM
        """Record the 3 capture layers and build the vLLM engine around them."""
        self._aux_layer_ids = list(aux_layer_ids)
        self._build_llm()

    def _build_llm(self) -> None:  # pragma: no cover - requires GPU + vLLM
        from vllm import LLM
        from vllm.config.kv_transfer import KVTransferConfig

        os.makedirs(self._shared_storage_path, exist_ok=True)
        # Capture the three EAGLE-3 layers plus the final pre-norm hidden
        # (layer id == num_hidden_layers) used to rebuild logits.
        #
        # Convention shift: ``HFEagle3TargetModel`` (and SGLang, matched to it)
        # hook the *output* of decoder layer ``aux_layer_id`` (== HF
        # ``hidden_states[aux_layer_id + 1]``), whereas vLLM's capture id ``k``
        # records the residual-stream value *entering* layer ``k`` (== the output
        # of layer ``k - 1``). Shift the aux ids by +1 so the vLLM backend captures
        # the exact same hidden states as the co-located HF backend, keeping the
        # supervision numerically equivalent across engines.
        capture_ids = [layer_id + 1 for layer_id in self._aux_layer_ids] + [self._num_layers]
        self._llm = LLM(
            model=self._model_path,
            trust_remote_code=self._trust_remote_code,
            dtype=vllm_dtype_str(self._dtype),
            tensor_parallel_size=self._tp_size,
            gpu_memory_utilization=self._gpu_memory_utilization,
            enforce_eager=True,
            enable_chunked_prefill=False,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": capture_ids}},
            },
            kv_transfer_config=KVTransferConfig(
                kv_connector="ExampleHiddenStatesConnector",
                kv_role="kv_producer",
                kv_connector_extra_config={"shared_storage_path": self._shared_storage_path},
            ),
            **self._vllm_kwargs,
        )
        logger.info("built vLLM extract_hidden_states engine; capture layers %s", capture_ids)

    @torch.no_grad()
    def forward_eagle3(  # pragma: no cover - requires GPU + vLLM
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one prefill per row and return ``(logits, aux_hidden_states)``.

        ``logits`` is ``[batch, seq, vocab]`` (full vocab, unshifted) rebuilt from
        the captured final hidden via the target's final RMSNorm + LM head;
        ``aux_hidden_states`` is ``[batch, seq, 3 * hidden]`` (the three capture
        layers concatenated, unshifted). Sequences must share a length (training
        batches are right-padded); ``attention_mask`` is unused because each row
        is a full causal prefill in vLLM (trailing pad tokens do not affect
        earlier positions under causal attention).
        """
        from vllm import SamplingParams
        from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector as hsc

        del attention_mask  # each row is a full causal prefill in vLLM
        if self._llm is None:
            raise RuntimeError("set_aux_layers must be called before forward_eagle3 (it builds the vLLM engine).")
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        prompts = [{"prompt_token_ids": input_ids[i].tolist()} for i in range(batch_size)]
        # ``LLM.generate`` returns outputs in input order.
        outputs = self._llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0))

        aux_rows: list[torch.Tensor] = []
        final_rows: list[torch.Tensor] = []
        for output in outputs:
            path = output.kv_transfer_params.get("hidden_states_path")
            obj = hsc.load_hidden_states(path)
            hidden = obj["hidden_states"]
            if not torch.is_tensor(hidden):
                hidden = torch.as_tensor(hidden)
            hsc.cleanup_hidden_states(path)
            # hidden: [seq, num_capture_layers, hidden]; layers [0:3] are EAGLE-3 aux, [3] is the final hidden.
            aux_rows.append(hidden[:, :3, :].reshape(hidden.shape[0], -1))
            final_rows.append(hidden[:, 3, :])

        aux = torch.stack(aux_rows, dim=0).to(device=device)
        final_hidden = torch.stack(final_rows, dim=0).to(device=device)
        logits = self._compute_logits(final_hidden)
        return logits, aux

    def _compute_logits(self, final_hidden: torch.Tensor) -> torch.Tensor:  # pragma: no cover - requires GPU + vLLM
        """Rebuild full-vocab logits from the captured pre-norm final hidden state."""
        if self._norm_w is None:
            self._embed_w, self._norm_w, self._lm_head_w = _load_target_head_weights(
                self._model_path, final_hidden.device
            )
        x = final_hidden.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self._rms_eps)
        x = x * self._norm_w.float()
        return x @ self._lm_head_w.float().t()

    def input_embedding_weight(self) -> torch.Tensor:  # pragma: no cover - requires GPU + vLLM
        """Return the target input-embedding weight ``[vocab, hidden]``."""
        if self._embed_w is None:
            self._embed_w, self._norm_w, self._lm_head_w = _load_target_head_weights(
                self._model_path, torch.device("cuda", torch.cuda.current_device())
            )
        return self._embed_w

    def close(self) -> None:  # pragma: no cover - requires GPU + vLLM
        """Release the vLLM engine (best effort)."""
        llm, self._llm = self._llm, None
        del llm
