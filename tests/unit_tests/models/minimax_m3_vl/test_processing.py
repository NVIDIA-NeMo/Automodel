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

"""Unit tests for ``build_minimax_m3_vl_processor``.

The helper loads the processor via ``AutoProcessor.from_pretrained`` and, because
the remote-code ``MiniMaxVLProcessor.__init__`` drops the loaded ``chat_template``,
reattaches it from the checkpoint's ``chat_template.jinja``. These tests mock
``AutoProcessor`` (no download / no remote code) and cover all three branches:
template already present, reattached from the .jinja, and left ``None`` when the
file is absent.
"""

from types import SimpleNamespace
from unittest import mock

from nemo_automodel.components.models.minimax_m3_vl.processing import build_minimax_m3_vl_processor

_PROCESSING = "nemo_automodel.components.models.minimax_m3_vl.processing.AutoProcessor"


@mock.patch(_PROCESSING)
def test_keeps_existing_chat_template(mock_auto_processor, tmp_path):
    """When the loaded processor already has a chat_template, it's returned untouched
    and the .jinja is not consulted (even if one exists)."""
    proc = SimpleNamespace(chat_template="EXISTING-TEMPLATE")
    mock_auto_processor.from_pretrained.return_value = proc
    # A .jinja present must be ignored because chat_template is not None.
    (tmp_path / "chat_template.jinja").write_text("FROM-FILE", encoding="utf-8")

    out = build_minimax_m3_vl_processor(str(tmp_path), trust_remote_code=True)

    assert out is proc
    assert out.chat_template == "EXISTING-TEMPLATE"
    mock_auto_processor.from_pretrained.assert_called_once_with(str(tmp_path), trust_remote_code=True)


@mock.patch(_PROCESSING)
def test_reattaches_chat_template_from_jinja(mock_auto_processor, tmp_path):
    """When chat_template is None and a chat_template.jinja exists, it's read in."""
    proc = SimpleNamespace(chat_template=None)
    mock_auto_processor.from_pretrained.return_value = proc
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}-TEMPLATE-BODY", encoding="utf-8")

    out = build_minimax_m3_vl_processor(str(tmp_path))

    assert out.chat_template == "{{ messages }}-TEMPLATE-BODY"


@mock.patch(_PROCESSING)
def test_leaves_none_when_no_jinja(mock_auto_processor, tmp_path):
    """When chat_template is None and no .jinja exists, it stays None (no error)."""
    proc = SimpleNamespace(chat_template=None)
    mock_auto_processor.from_pretrained.return_value = proc

    out = build_minimax_m3_vl_processor(str(tmp_path))

    assert out.chat_template is None
