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

import os

from transformers import AutoProcessor
from transformers.processing_utils import ProcessorMixin


def build_minimax_m3_vl_processor(pretrained_model_name_or_path: str, **kwargs) -> ProcessorMixin:
    """Load the MiniMax M3 VL processor with its chat template attached.

    The checkpoint ships a ``chat_template.jinja``, but the remote-code
    ``MiniMaxVLProcessor.__init__`` declares only ``(image_processor, tokenizer,
    video_processor, **kwargs)`` and does not forward ``chat_template`` to
    ``ProcessorMixin``. Transformers' ``from_args_and_dict`` only passes the
    constructor's *explicit* named arguments, so the loaded template is
    classified as an unused kwarg and dropped, leaving ``processor.chat_template``
    as ``None`` and ``apply_chat_template`` raising "does not have a chat
    template". Reattach it from the checkpoint's ``chat_template.jinja``.

    Args:
        pretrained_model_name_or_path: Local path (or hub id) of the checkpoint.
        **kwargs: Forwarded to ``AutoProcessor.from_pretrained``
            (e.g. ``trust_remote_code=True``).

    Returns:
        The processor with a non-``None`` ``chat_template`` when one is available.
    """
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
    if getattr(processor, "chat_template", None) is None:
        template_path = os.path.join(pretrained_model_name_or_path, "chat_template.jinja")
        if os.path.isfile(template_path):
            with open(template_path, encoding="utf-8") as f:
                processor.chat_template = f.read()
    return processor
