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

"""Hand JSON-string tool-call arguments to a chat template as dicts.

Some chat templates render every tool-call argument as its own field, so they need ``function.arguments`` as a
mapping. Automodel's chat datasets store arguments as JSON strings (the OpenAI wire format, and the only form
Arrow-backed datasets keep intact). ``wrap_tokenizer`` makes a tokenizer instance's ``apply_chat_template`` parse
such strings back into dicts first, which is also what the serving stacks pass to those templates.
``DictToolArgumentsTokenizer`` is registered in ``TokenizerRegistry`` for the model types whose templates need it
(Beam, DeepSeek-V4.1, Nemotron-H); other models are unaffected, and the tokenizer class (and hence ``save_pretrained`` output)
is unchanged.
"""

import functools
import json
import os
from typing import Any

from transformers import PreTrainedTokenizerBase


def _message_with_dict_arguments(message: Any) -> Any:
    """Return ``message`` with JSON-object string arguments parsed; the input is never mutated."""
    if not isinstance(message, dict) or not isinstance(message.get("tool_calls"), list):
        return message
    tool_calls = []
    changed = False
    for tool_call in message["tool_calls"]:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = None
            # Anything that is not a JSON object stays as is, so the template raises exactly as before.
            if isinstance(parsed, dict):
                tool_call = {**tool_call, "function": {**function, "arguments": parsed}}
                changed = True
        tool_calls.append(tool_call)
    return {**message, "tool_calls": tool_calls} if changed else message


def tool_call_arguments_as_dicts(conversation: Any) -> Any:
    """Parse JSON-string tool-call arguments into dicts for one conversation or a batch of conversations.

    Args:
        conversation: A list of chat messages, a list of such lists, or anything else ``apply_chat_template``
            accepts.

    Returns:
        The same structure with every tool call whose ``function.arguments`` is a JSON-object string replaced by
        a copy holding the parsed dict. Other arguments, messages and inputs are returned unchanged, and the input
        is never mutated.
    """
    if not isinstance(conversation, (list, tuple)):
        return conversation
    if conversation and isinstance(conversation[0], (list, tuple)):
        return [tool_call_arguments_as_dicts(c) for c in conversation]
    return [_message_with_dict_arguments(m) for m in conversation]


def _apply_chat_template_with_dict_arguments(tokenizer, conversation, *args, **kwargs):
    # Call the class-level method: the instance attribute is this wrapper.
    return type(tokenizer).apply_chat_template(tokenizer, tool_call_arguments_as_dicts(conversation), *args, **kwargs)


def wrap_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Make ``tokenizer.apply_chat_template`` accept JSON-string tool-call arguments.

    The wrapper is an instance attribute, so the tokenizer class (and its ``save_pretrained`` output and
    pickling) stays the one the default loader picked. A ``functools.partial`` over a module-level function,
    not a closure, keeps the tokenizer picklable and deep-copyable.

    Args:
        tokenizer: The tokenizer to wrap in place.

    Returns:
        The same tokenizer, whose ``apply_chat_template`` now passes conversations through
        :func:`tool_call_arguments_as_dicts` first.
    """
    tokenizer.apply_chat_template = functools.partial(_apply_chat_template_with_dict_arguments, tokenizer)
    return tokenizer


class DictToolArgumentsTokenizer:
    """Tokenizer factory for models whose chat template needs tool-call arguments as dicts."""

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike[str], *args: object, **kwargs: object
    ) -> PreTrainedTokenizerBase:
        """Load the tokenizer exactly as the default path does, then wrap ``apply_chat_template``.

        Args:
            pretrained_model_name_or_path: Hub id or local directory of the tokenizer.
            *args: Forwarded to the default loader.
            **kwargs: Forwarded to the default loader.

        Returns:
            The default Automodel tokenizer with :func:`wrap_tokenizer` applied.
        """
        from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import (
            NeMoAutoTokenizerWithBosEosEnforced,
        )

        tokenizer = NeMoAutoTokenizerWithBosEosEnforced.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return wrap_tokenizer(tokenizer)
