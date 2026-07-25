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

"""HF cache resolution helpers for the diffusers bridge.

Diffusers' ``from_pretrained`` resolves a bare repo id in *online* mode by
default: it issues per-file network requests to the Hub to revalidate ETags
before deciding whether to reuse the cache. Even a warm ``HF_HOME`` is then
re-validated over the network, and any ETag drift or partial cache turns into a
fresh download. The transformers bridge avoids this by pre-resolving the repo
to a local snapshot directory (see ``_resolve_model_dir`` in
``nemo_automodel/_transformers/model_init.py``) and handing that directory to
HF, which then does zero network I/O. This module ports the same discipline to
the diffusion path.
"""

import os

from nemo_automodel.shared.import_utils import safe_import

HF_HUB_AVAILABLE, _ = safe_import("huggingface_hub")

if HF_HUB_AVAILABLE:
    from huggingface_hub import snapshot_download
else:
    snapshot_download = None


def resolve_diffusion_model_dir(model_id: str) -> str:
    """Resolve a HF repo id to a local snapshot directory.

    Mirrors the transformers bridge so a warm ``HF_HOME`` is never re-validated
    over the network. Local paths are returned unchanged. For repo ids, the
    snapshot is downloaded once only when the cache is cold and the process is
    online (``HF_HUB_OFFLINE`` unset); the returned directory is then resolved
    with ``local_files_only=True`` so the subsequent ``from_pretrained`` call
    performs no network I/O.

    Args:
        model_id: A HuggingFace repo id or a local filesystem path.

    Returns:
        A local directory path containing the model snapshot. When
        ``huggingface_hub`` is unavailable, ``model_id`` is returned unchanged
        so resolution falls back to HF's own handling.
    """
    if os.path.isdir(model_id) or not HF_HUB_AVAILABLE:
        return model_id

    if os.environ.get("HF_HUB_OFFLINE", "0") != "1":
        # Cold cache + online: fetch the snapshot once.
        snapshot_download(model_id)
    # Resolve (and require) the local snapshot without revalidating over the network.
    return snapshot_download(model_id, local_files_only=True)
