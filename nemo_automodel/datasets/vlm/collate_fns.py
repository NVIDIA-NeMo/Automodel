from typing import Dict
import torch
from nemo_automodel.datasets.vlm.utils import extract_skipped_token_ids
from qwen_vl_utils import process_vision_info


def qwen2_5_collate_fn(examples: list, processor) -> Dict[str, torch.Tensor]:
    """Collate function for Qwen2.5 VL model."""
    skipped_tokens = extract_skipped_token_ids(processor)

    # examples = preprocess_examples(examples)
    texts = [
        processor.apply_chat_template(example["conversation"], tokenize=False)
        for example in examples
    ]
    image_inputs = [
        process_vision_info(example["conversation"])[0] for example in examples
    ]

    batch = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels

    return batch


def gemma3_collate_fn(examples: list, processor) -> Dict[str, torch.Tensor]:
    """Collate function for Gemma3 model."""
    skipped_tokens = extract_skipped_token_ids(processor)

    # examples = preprocess_examples(examples)
    batch = processor.apply_chat_template(
        [example["conversation"] for example in examples],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )

    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels

    return batch


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "Gemma3Processor": gemma3_collate_fn,
}
