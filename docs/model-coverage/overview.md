# Overview

NeMo AutoModel integrates with Hugging Face `transformers`. As a result, any LLM or VLM that can be instantiated through `transformers` can also be used via NeMo AutoModel, subject to runtime, third-party software dependencies and feature compatibility.

## Version compatibility and Day‑0 support

- AutoModel closely tracks the latest `transformers` version and updates its dependency on a regular basis.
- Newly released Hugging Face models may require a more recent `transformers` version than what NeMo currently supports; those models will become available in NeMo AutoModel after the next `transformers` version bump.
- We are working on introducing a continuous integration (CI) pipeline that will automatically bump the supported `transformers` version as soon as a new release is detected. This will enable even faster support for the newest Hugging Face models.

**Note:** To use newly released models, you may need to upgrade your NeMo AutoModel installation. This process is similar to upgrading `transformers` itself to access the latest model support. In practice, the upgrade behavior and familiar `Auto*` APIs mirror `transformers`, with AutoModel additionally providing optional performance accelerations and distributed training features.

## Extending model support with the custom registry

AutoModel includes a custom model registry that allows teams to:

- Add custom implementations to extend support to models not yet covered upstream.
- Provide optimized, extended or faster implementations for specific models while retaining the same AutoModel interface.

## Supported Hugging Face Auto classes

| Auto class                          | Task                     | Status     | Notes                                     |
|-------------------------------------|--------------------------|------------|-------------------------------------------|
| `AutoModelForCausalLM`              | Text Generation (LLM)    | Supported  | See [`docs/model-coverage/llm.md`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/llm.md).         |
| `AutoModelForImageTextToText`       | Image-Text-to-Text (VLM) | Supported  | See [`docs/model-coverage/vlm.md`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/vlm.md).         |
| `AutoModelForSequenceClassification`| Sequence Classification  | WIP        | Early support; interfaces may change.     |


## When a model listed on Hugging Face Hub may not be supported

Sometimes a model listed on the Hugging Face Hub may not support finetuning in NeMo AutoModel.
If you encounter any such model, please open a [GitHub issue](https://github.com/NVIDIA-NeMo/Automodel/issues) requesting support by sharing the model-id of interest as well as any stack trace you see. We summarize here some cases (non-exhaustive):

| Issue                              | Example Error Message    | Solution                                    |
|------------------------------------|--------------------------|---------------------------------------------|
|Model has explicitly disabled training functionality in the model-definition code. | — | Make the model available via our custom registry. Please open a new GitHub issue, requesting support. |
| Model requires newer transformers version | The checkpoint you are trying to load has model type `deepseek_v32` but Transformers does not recognize this architecture. | Upgrade the transformers version you use, and/or open a new GitHub issue. |
| Model upper-bounds transformer version, requiring older version | — | Open a new GitHub issue. |
| Back/stale Checkpoint | OSError: `meta-llama/Llama-2-70b` does not appear to have a file named pytorch_model.bin, model.safetensors, tf_model.h5, model.ckpt or flax_model.msgpack. | Open a new GitHub issue. |

These cases typically stem from upstream packaging or dependency constraints and would surface similarly when using `transformers` directly. AutoModel mirrors the familiar load/finetune semantics.

If you encounter any issue, you can try:

- Upgrade to a NeMo AutoModel release that supports the required `transformers` version.
- If the model uses custom code, set `trust_remote_code=True` when loading.
- Open an issue with the model-id and error for us to prioritize support or add a registry-backed implementation.
