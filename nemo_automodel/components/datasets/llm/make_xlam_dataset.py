import json
import logging
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset

from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    _package_tokenized_example,
)


def _normalize_type(param_type: str) -> str:
    param_type = (param_type or "").strip()

    if "," in param_type and "default" in param_type:
        param_type = param_type.split(",")[0].strip()

    if param_type.startswith("default="):
        return "string"

    param_type = param_type.replace(", optional", "").strip()

    if param_type.startswith("Callable"):
        return "string"
    if param_type.startswith("Tuple"):
        return "array"
    if param_type.startswith("List["):
        return "array"
    if param_type.startswith("Set") or param_type == "set":
        return "array"

    type_mapping: Dict[str, str] = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "List": "array",
        "Dict": "object",
        "set": "array",
        "Set": "array",
    }
    return type_mapping.get(param_type, "string")


def _convert_tools_to_openai_spec(tools: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Convert XLAM `tools` field to OpenAI function-spec tools list.
    Accepts either a JSON string or a parsed list.
    """
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse tools JSON: {e}")
            return []

    if not isinstance(tools, list):
        logging.warning(f"Expected tools to be a list, got {type(tools)}")
        return []

    openai_tools: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            logging.warning(f"Expected tool to be a dict, got {type(tool)}")
            continue
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            logging.warning("Expected tool['parameters'] to be a dict; skipping tool")
            continue

        normalized_parameters: Dict[str, Dict[str, Any]] = {}
        for param_name, param_info in parameters.items():
            if not isinstance(param_info, dict):
                logging.warning(
                    f"Expected parameter info to be a dict for '{param_name}', got {type(param_info)}"
                )
                continue

            param_dict = {
                "description": param_info.get("description", ""),
                "type": _normalize_type(param_info.get("type", "")),
            }
            default_value = param_info.get("default")
            if default_value is not None and default_value != "":
                param_dict["default"] = default_value
            normalized_parameters[param_name] = param_dict

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": {"type": "object", "properties": normalized_parameters},
                },
            }
        )
    return openai_tools


def _convert_tool_calls(xlam_tools: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Convert XLAM `answers` (tool calls) to OpenAI-style tool call entries for assistant messages.
    Accepts JSON string or parsed list.
    """
    if isinstance(xlam_tools, str):
        try:
            xlam_tools = json.loads(xlam_tools)
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse answers JSON: {e}")
            return []
    if not isinstance(xlam_tools, list):
        return []

    tool_calls: List[Dict[str, Any]] = []
    for tool in xlam_tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        arguments = tool.get("arguments", {})
        tool_calls.append({"type": "function", "function": {"name": name, "arguments": arguments}})
    return tool_calls


def _formatting_xlam_with_chat_template(
    example: Dict[str, Any],
    tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    seq_length: Optional[int] = None,
    start_of_turn_token: Optional[str] = None,
):
    """
    Format XLAM example into chat template with tools, using tokenizer.apply_chat_template(tool=...).
    The assistant turn contains tool_calls for supervised training.
    """
    query = example.get("query", "") or ""
    tools_spec = _convert_tools_to_openai_spec(example.get("tools", []))
    tool_calls = _convert_tool_calls(example.get("answers", []))

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": query},
        # Supervised target: include assistant tool calls; content may be empty string
        {"role": "assistant", "content": "", "tool_calls": tool_calls} if tool_calls else {"role": "assistant", "content": ""},
    ]

    input_ids = tokenizer.apply_chat_template(messages, tools=tools_spec)

    if isinstance(start_of_turn_token, str):
        start_of_turn_token_id = tokenizer(start_of_turn_token, add_special_tokens=False)["input_ids"][0]
        first_start = input_ids.index(start_of_turn_token_id)
        response_start = input_ids.index(start_of_turn_token_id, first_start + 1)
    else:
        response_start = 0

    return _package_tokenized_example(True, input_ids, eos_token_id, pad_token_id, seq_length, response_start)


def _is_single_tool_call_example(example: Dict[str, Any]) -> bool:
    answers = example.get("answers")
    try:
        calls = json.loads(answers) if isinstance(answers, str) else answers
    except Exception:
        return False
    return isinstance(calls, list) and len(calls) == 1


def make_xlam_dataset(
    tokenizer,
    seq_length: Optional[int] = None,
    limit_dataset_samples: Optional[int] = None,
    start_of_turn_token: Optional[str] = None,
    split: str = "train",
    dataset_name: str = "Salesforce/xlam-function-calling-60k",
    dataset_type: str = "single",
):
    """
    Load and preprocess the XLAM function-calling dataset for SFT with function calling.

    - Builds messages from `query` and `answers` (as assistant `tool_calls`).
    - Passes available tools to the tokenizer via `apply_chat_template(tool=...)`.
    - Packages tokenized examples with answer-only loss masking.
    """
    if limit_dataset_samples is not None:
        assert isinstance(limit_dataset_samples, int), "Expected limit_dataset_samples to be an int"
        if "[" not in split:
            split = f"{split}[:{limit_dataset_samples}]"
        else:
            logging.warning(f"Dataset split {split} already has a slice, skipping limit_dataset_samples")

    dataset = load_dataset(dataset_name, split=split)

    if dataset_type == "single":
        dataset = dataset.filter(_is_single_tool_call_example)

    # Tokenization configuration
    eos_token_id = getattr(tokenizer, "eos_token_id", 0)
    pad_token_id = _add_pad_token(tokenizer) or eos_token_id

    fmt_fn = lambda x: _formatting_xlam_with_chat_template(
        x, tokenizer, eos_token_id, pad_token_id, seq_length, start_of_turn_token
    )

    # Map and drop original columns
    return dataset.map(
        fmt_fn,
        batched=False,
        remove_columns=dataset.column_names,
    )


