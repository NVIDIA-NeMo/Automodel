### Overview

NeMo AutoModel integrates with Hugging Face `transformers`. As a result, any LLM or VLM that can be instantiated through `transformers` can also be used via NeMo AutoModel, subject to runtime, third-party software dependencies and feature compatibility.

### Version compatibility and Day‑0 support

- AutoModel tracks the latest `transformers` version that is officially supported within NeMo.
- Newly released Hugging Face models may depend on a newer `transformers` version than NeMo currently supports. In such cases, those models are not available in AutoModel until NeMo updates its supported `transformers` version.

**Note:** To use newly released models, you may need to upgrade the `transformers` dependency to a version that supports those models and is compatible with NeMo.

### Extending model support with the custom registry

AutoModel includes a custom model registry that allows teams to:

- Add custom implementations to extend support to models not yet covered upstream.
- Provide optimized or faster implementations for specific models while retaining the same AutoModel interface.

### Supported Hugging Face Auto classes

| Auto class                          | Task                     | Status     | Notes                                     |
|-------------------------------------|--------------------------|------------|-------------------------------------------|
| `AutoModelForCausalLM`              | Text Generation (LLM)    | Supported  | See `docs/model-coverage/llm.md`.         |
| `AutoModelForImageTextToText`       | Image-Text-to-Text (VLM) | Supported  | See `docs/model-coverage/vlm.md`.         |
| `AutoModelForSequenceClassification`| Sequence Classification  | WIP        | Early support; interfaces may change.     |


### When may a model from Hugging Face not be supported

There are cases where a model is available on the Hugging Face Hub, but you may not be able to finetune this model. We summarize here some cases (non-exhaustive):

| Issue                              | Example Error Message    | Solution                                    |
|------------------------------------|--------------------------|---------------------------------------------|
|Model has explicitly disabled training functionality in the model-definition code. || Make the model available via our custom registry. Please open a new GitHub issue, requesting support. |
| Model requires newer transformers version | The checkpoint you are trying to load has model type `deepseek_v32` but Transformers does not recognize this architecture. | Upgrade the transformers version you use, and/or open a new GitHub issue. |
