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

"""Distributed-parallelization registration for native MuseGlimmer."""

from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.common.tp_plan import gated_decoder_tp_plan


def _parallelize_muse_glimmer(model, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Shard the language backbone including its attention output gate; vision stays replicated."""
    if sequence_parallel:
        import warnings

        warnings.warn("sequence_parallel=True is not yet supported for MuseGlimmer and will be ignored.", stacklevel=2)
    plan = gated_decoder_tp_plan()
    plan["model.layers.*.self_attn.output_gate_proj"] = ColwiseParallel()
    return plan


class MuseGlimmerModelParallelizer(ModelParallelizer):
    """Apply standard dense parallelism and install the model-owned CP mesh."""

    tp_plan = staticmethod(_parallelize_muse_glimmer)

    def _apply(self, model, device_mesh, **kwargs):
        tp_mesh = device_mesh["tp"] if "tp" in device_mesh.mesh_dim_names else None
        tp_size = tp_mesh.size() if tp_mesh is not None else 1
        cp_mesh = device_mesh["cp"] if "cp" in device_mesh.mesh_dim_names else None
        cp_size = cp_mesh.size() if cp_mesh is not None else 1
        num_kv_heads = model.config.num_key_value_heads
        if tp_size > num_kv_heads or num_kv_heads % tp_size != 0:
            raise ValueError(
                f"MuseGlimmer supports TP1 or TP2 because it has {num_kv_heads} KV heads; got tp_size={tp_size}."
            )
        result = super()._apply(model, device_mesh, **kwargs)
        model.cp_mesh = cp_mesh if cp_mesh is not None and cp_mesh.size() > 1 else None
        model.model.cp_mesh = model.cp_mesh
        # The generic dense-TE pass configures every CP run. TP-only native
        # MuseGlimmer is BSHD-capable (not THD-only), so keep that model-specific
        # setup here rather than broadening the generic infrastructure gate.
        if tp_size > 1 and cp_size <= 1 and model.backend.attn == "te":
            from nemo_automodel.components.distributed.context_parallel.utils import attach_te_context_parallel

            configured = attach_te_context_parallel(result, None, tp_mesh)
            if configured != len(model.model.layers):
                raise ValueError(
                    "MuseGlimmer TP selected Transformer Engine attention, but only "
                    f"{configured}/{len(model.model.layers)} attention modules were configured."
                )
        return result


PARALLELIZER = MuseGlimmerModelParallelizer()

__all__ = ["PARALLELIZER"]
