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

"""Launch a remote EAGLE-3 target server (train-inference disaggregation).

Loads the frozen target model on this process's GPU and serves draft-vocab
supervision (aux hidden states, ``target_probs``, ``position_mask``) to a
training client over HTTP (control plane) + NCCL (data plane).

Typical usage (single-GPU server)::

    CUDA_VISIBLE_DEVICES=0 python -m nemo_automodel.components.speculative.serve_target \\
        --target meta-llama/Llama-3.1-8B-Instruct \\
        --host 0.0.0.0 --port 8001

Then point training at it::

    recipe_args.target_model_backend: remote
    recipe_args.remote_urls: ["http://<server-host>:8001"]
    recipe_args.target_prefetch_depth: 1

Verify readiness with ``curl http://<host>:8001/health``. NCCL GPU-direct
transfer requires sglang installed in the server's environment; without it the
server transparently falls back to the binary wire format.
"""

from __future__ import annotations

import argparse
import logging

import torch

from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
from nemo_automodel.components.speculative.eagle.remote.server import TargetModelServer, serve
from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel

logger = logging.getLogger(__name__)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an EAGLE-3 target model for remote draft training.")
    parser.add_argument("--target", required=True, help="Target model name or path.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (0.0.0.0 for cross-machine).")
    parser.add_argument("--port", type=int, default=8001, help="HTTP control-plane port.")
    parser.add_argument(
        "--nccl-port",
        type=int,
        default=None,
        help="NCCL rendezvous port (default: HTTP port + 100).",
    )
    parser.add_argument(
        "--aux-layer-ids",
        type=int,
        nargs="+",
        default=None,
        help="Three aux hidden-state layer indices (default: EAGLE-3 low/mid/high recipe).",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code when loading the target.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Load the target model and run the blocking HTTP + NCCL server."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    logger.info("Loading target model %s on %s", args.target, device)
    target_model = NeMoAutoModelForCausalLM.from_pretrained(
        args.target,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    target_model.to(device)
    target_model.requires_grad_(False)
    target_wrapper = HFEagle3TargetModel(target_model, aux_layer_ids=args.aux_layer_ids)

    nccl_port = args.nccl_port if args.nccl_port is not None else args.port + 100
    server_logic = TargetModelServer(target_wrapper, nccl_port=nccl_port, host=args.host)
    serve(server_logic, args.host, args.port)


if __name__ == "__main__":
    main()
