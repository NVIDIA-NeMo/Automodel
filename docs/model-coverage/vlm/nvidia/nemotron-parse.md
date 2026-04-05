# Nemotron-Parse

[Nemotron-Parse-v1.1](https://huggingface.co/nvidia/Nemotron-Parse-v1.1) is NVIDIA's document parsing VLM, specializing in extracting structured information from complex documents including tables, forms, and mixed-content PDFs.

:::{card}
| | |
|---|---|
| **Task** | Document Parsing |
| **Architecture** | `NemotronParseForConditionalGeneration` |
| **Parameters** | varies |
| **HF Org** | [nvidia](https://huggingface.co/nvidia) |
:::

## Available Models

- **Nemotron-Parse-v1.1**

## Architecture

- `NemotronParseForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Nemotron-Parse v1.1 | `nvidia/Nemotron-Parse-v1.1` |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`nemotron_parse_v1_1.yaml <../../../examples/vlm_finetune/nemotron/nemotron_parse_v1_1.yaml>` | cord-v2 | SFT — Nemotron-Parse on CORD-v2 |

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [nvidia/Nemotron-Parse-v1.1](https://huggingface.co/nvidia/Nemotron-Parse-v1.1)
