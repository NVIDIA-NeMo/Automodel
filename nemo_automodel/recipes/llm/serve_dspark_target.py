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

"""Serve a frozen DSpark target to a remote training run (train-inference disaggregation).

It reads the same YAML as ``train_dspark`` (``recipe_args`` selects the target and
the captured layers), listens on ``target_server.host`` / ``target_server.port``
(default ``0.0.0.0:8001``), and runs the target in this single process::

    python -m nemo_automodel.recipes.llm.serve_dspark_target \\
        -c examples/speculative/dspark/qwen3_0.6b_dspark_remote.yaml

See "Target Backends" in ``examples/speculative/README.md`` for the training side.
"""

from __future__ import annotations

import logging
import os

import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.speculative.dspark.remote import DSparkTargetModelServer
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel
from nemo_automodel.components.speculative.dspark.target_utils import read_target_model_type
from nemo_automodel.components.speculative.eagle.remote import protocol
from nemo_automodel.components.speculative.eagle.remote.server import serve
from nemo_automodel.recipes.llm._dspark_target_build import (
    build_dspark_target,
    resolve_dspark_target_layer_ids,
    validate_dspark_parallelism_axes,
)

logger = logging.getLogger(__name__)


def build_server(cfg, *, host: str, nccl_port: int) -> DSparkTargetModelServer:
    """Load the frozen target described by ``cfg`` and wrap it in the request-handling logic.

    Args:
        cfg: A ``train_dspark`` config; ``recipe_args`` selects the target and the
            captured layers.
        host: Bind/advertise address, also the NCCL rendezvous master.
        nccl_port: TCP rendezvous port for the NCCL data plane.

    Returns:
        The server logic, ready for :func:`serve`.

    Raises:
        NotImplementedError: When launched with more than one rank.
    """
    validate_dspark_parallelism_axes(cfg)
    dist_env = initialize_distributed(
        backend=cfg.get("dist_env", {}).get("backend", "nccl"),
        timeout_minutes=cfg.get("dist_env", {}).get("timeout_minutes", 30),
    )
    if dist_env.world_size > 1:
        raise NotImplementedError(
            f"serve_dspark_target runs the target in a single process, but was launched with "
            f"world_size={dist_env.world_size}. Multi-rank serving of expert-parallel targets is "
            "not supported yet (issue #4325)."
        )
    device = dist_env.device or torch.device("cpu")
    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    recipe_cfg = cfg.recipe_args
    trust_remote_code = bool(recipe_cfg.get("trust_remote_code", False))
    target_path = recipe_cfg.target_model_name_or_path
    target_text_config, target_model = build_dspark_target(
        cfg=cfg,
        recipe_cfg=recipe_cfg,
        world_size=dist_env.world_size,
        device=device,
        compute_dtype=compute_dtype,
        model_type=read_target_model_type(target_path, trust_remote_code),
        target_path=target_path,
        trust_remote_code=trust_remote_code,
    )
    target_model.requires_grad_(False)
    target_wrapper = HFDSparkTargetModel(
        target_model,
        target_layer_ids=resolve_dspark_target_layer_ids(recipe_cfg, int(target_text_config.num_hidden_layers)),
    )
    server = DSparkTargetModelServer(target_wrapper, target_text_config, nccl_port=nccl_port, host=host)
    logger.info("Serving DSpark target %s on %s (%s)", target_path, device, server.target_info)
    return server


def main(config_path: str | None = None) -> None:
    """CLI entry point. Parses ``-c <config.yaml>``, loads the target, and serves until a client disconnects."""
    logging.basicConfig(
        level=getattr(logging, os.environ.get("DSPARK_SERVE_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse_args_and_load_config(config_path)
    host = str(cfg.get("target_server.host", "0.0.0.0"))
    port = int(cfg.get("target_server.port", 8001))
    # The client picks the NCCL rendezvous port (HTTP port + offset, or NEMO_EAGLE_NCCL_PORT)
    # and sends it with its init request; this is only the server-side fallback.
    serve(build_server(cfg, host=host, nccl_port=port + protocol.NCCL_PORT_OFFSET), host, port)


if __name__ == "__main__":
    main()
