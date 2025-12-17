import json
import logging
from typing import Dict, List, Optional

from datasets import load_dataset

from nemo_automodel.components.datasets.llm.formatting_utils import _add_pad_token, format_chat_template

logger = logging.getLogger(__name__)

# Map lightweight xLAM types to JSON schema / OpenAI tool types
_TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "object": "object",
}


def _json_load_if_str(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _convert_tools(raw_tools: List[Dict]) -> List[Dict]:
    """
    Convert xLAM tool definitions into OpenAI tool schema.
    """
    converted = []
    for tool in raw_tools or []:
        name = tool.get("name")
        if not name:
            logger.warning("Skipping tool without name: %s", tool)
            continue
        description = tool.get("description", "")
        params_raw: Dict = tool.get("parameters") or {}

        properties: Dict[str, Dict] = {}
        required: List[str] = []
        for param_name, param_def in params_raw.items():
            if param_def is None:
                continue
            param_type = str(param_def.get("type", "string")).lower()
            mapped_type = _TYPE_MAP.get(param_type, "string")
            prop = {"type": mapped_type, "description": param_def.get("description", "")}
            if "enum" in param_def:
                prop["enum"] = param_def["enum"]
            if "default" in param_def:
                prop["default"] = param_def["default"]
            else:
                required.append(param_name)
            properties[param_name] = prop

        parameters_schema: Dict = {"type": "object", "properties": properties}
        if required:
            parameters_schema["required"] = required

        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters_schema,
                },
            }
        )
    return converted


def _convert_tool_calls(raw_calls: List[Dict], example_id: Optional[int] = None) -> List[Dict]:
    """
    Convert xLAM answers field into OpenAI tool_calls messages.
    """
    tool_calls = []
    for idx, call in enumerate(raw_calls or []):
        name = call.get("name")
        if not name:
            logger.warning("Skipping call without name: %s", call)
            continue
        arguments = call.get("arguments", "")

        call_id = f"call_{example_id}_{idx}" if example_id is not None else f"call_{idx}"
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    # Keep arguments as JSON string per OpenAI tool calling format.
                    "arguments": arguments,
                },
            }
        )
    return tool_calls


def _format_example(
    example,
    tokenizer,
    eos_token_id,
    pad_token_id,
    seq_length=None,
    padding=None,
    truncation=None,
):
    tools = _convert_tools(_json_load_if_str(example["tools"]))
    tool_calls = _convert_tool_calls(_json_load_if_str(example["answers"]), example_id=example.get("id"))

    formatted_text = [
        {"role": "user", "content": example["query"]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls,
        },
    ]

    return format_chat_template(
        tokenizer=tokenizer,
        formatted_text=formatted_text,
        tools=tools,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        seq_length=seq_length,
        padding=padding,
        truncation=truncation,
        answer_only_loss_mask=True,
    )


def make_xlam_dataset(
    tokenizer,
    seq_length=None,
    limit_dataset_samples=None,
    fp8=False,  # parity with other dataset builders
    split="train",
    dataset_name="Salesforce/xlam-function-calling-60k",
    padding=False,
    truncation=False,
):
    """
    Load and preprocess the xLAM function-calling dataset to OpenAI messages
    compatible with the bulbasaur chat template (tool-calling aware).

    Each example is formatted as:
      - user: the natural language query
      - assistant: emits tool_calls with serialized arguments
      - tools: OpenAI function schema derived from the dataset tool specs
    """
    if limit_dataset_samples is not None:
        assert isinstance(limit_dataset_samples, int), "Expected limit_dataset_samples to be an int"
        if "[" not in split:
            split = f"{split}[:{limit_dataset_samples}]"
        else:
            logger.warning("Dataset split %s already contains slice, skipping limit_dataset_samples", split)

    dataset = load_dataset(dataset_name, split=split)

    eos_token_id = getattr(tokenizer, "eos_token_id", 0)
    pad_token_id = _add_pad_token(tokenizer) or eos_token_id

    fmt_fn = lambda x: _format_example(  # noqa: E731
        x,
        tokenizer,
        eos_token_id,
        pad_token_id,
        seq_length,
        padding,
        truncation,
    )

    return dataset.map(
        fmt_fn,
        batched=False,
        remove_columns=["id", "query", "answers", "tools"],
    )
