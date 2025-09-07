# Knowledge Distillation with NeMo-AutoModel

This guide walks through fine-tuning a **student** LLM with the help of a
larger **teacher** model using the new `knowledge_distillation` recipe.

In particular, we will show how to distill a 3B (`meta-llama/Llama-3.2-3B`) model into a 1B (`meta-llama/Llama-3.2-1B`) model.

---

## 1. What is Knowledge Distillation?

Knowledge distillation (KD) transfers the *dark knowledge* of a high-capacity
teacher model to a smaller student by minimizing the divergence between their
predicted distributions.  The student learns from both the ground-truth labels
(Cross-Entropy loss, **CE**) and the soft targets of the teacher (Kullback-Leibler
loss, **KD**):


$$
  \mathcal{L} = (1-\alpha) \cdot \mathcal{L}_{\textrm{CE}}(p^{s}, y) + \alpha \cdot \mathcal{KL}(p^{s}, p^{t})
$$

where $\(\alpha\)$ is `kd_ratio`, $\(T\)$ is `temperature` and $y$ the labels. In addition, $p^{s} = softmax(z^{s}, T)$ and $p^{t} = softmax(z^{t}, T)$
---

## 2. Prepare the YAML config

A ready-to-use example is provided at
`examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml`.  Important sections:

* `model` – the student to be fine-tuned (1 B parameters in the example)
* `teacher_model` – a larger frozen model used for supervision (7 B)
* `kd_ratio` – blend between CE and KD loss
* `temperature` – softens probability distributions before KL-divergence
* `peft` – **optional** LoRA config (commented). Uncomment to train only a
  handful of parameters.

Feel free to tweak these values as required.

### Example YAML

```yaml
# Example config for knowledge distillation fine-tuning
# Run with:
#   automodel knowledge_distillation llm -c examples/llm/llama_3_2_1b_kd.yaml

step_scheduler:
  global_batch_size: 32
  local_batch_size: 1
  ckpt_every_steps: 200
  val_every_steps: 100  # will run every x number of gradient steps
  num_epochs: 2

dist_env:
  backend: nccl
  timeout_minutes: 1

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 1111
  ranked: true

# Student
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  torch_dtype: bf16

# Teacher
teacher_model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-3B
  torch_dtype: bf16

checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/
  model_save_format: safetensors
  save_consolidated: false

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  tp_size: 1
  cp_size: 1
  pp_size: 1
  sequence_parallel: false

# PEFT can be enabled by uncommenting below – student weights will remain small
# peft:
#   _target_: nemo_automodel.components._peft.lora.PeftConfig
#   match_all_linear: true
#   dim: 16
#   alpha: 32
#   use_triton: true

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

# Knowledge-distillation hyper-params
kd_ratio: 0.5          # 0 → pure CE, 1 → pure KD
kd_loss_fn:
  _target_: nemo_automodel.components.loss.kd_loss.KDLoss
  ignore_index: -100
  temperature: 1.0
  fp32_upcast: true

# Optimizer
optimizer:
  _target_: torch.optim.Adam
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 1.0e-5
  weight_decay: 0

# Dataset / Dataloader
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: false

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.hellaswag.HellaSwag
  path_or_dataset: rowan/hellaswag
  split: validation
  num_samples_limit: 64

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
```

### Current limitations

* Pipeline parallelism (`pp_size > 1`) is not yet supported – planned for a future release.
* Distilling Vision-Language models (`vlm` recipe) is currently not supported.
* Student and teacher models must share the same tokenizer for now; support for different tokenizers will be added in the future.

---

## 3. Launch training

### Single-GPU quick run

```bash
# Runs on a single device of the current host
automodel kd llm --nproc-per-node=1 -c examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml
```

### Multi-GPU (single node)

```bash
# Leverage all GPUs on the local machine
torchrun --nproc-per-node $(nvidia-smi -L | wc -l) \
    nemo_automodel/recipes/llm/kd.py \
    -c examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml
```

### SLURM cluster

The CLI seamlessly submits SLURM jobs when a `slurm` section is added to the
YAML.  Refer to `docs/guides/installation.md` for cluster instructions.

---

## 4. Monitoring

Metrics such as *train_loss*, *kd_loss*, *learning_rate* and *tokens/sec* are
logged to **WandB** when the corresponding section is enabled.

---

## 5. Checkpoints & Inference

- Checkpoints are written under the directory configured in the `checkpoint.checkpoint_dir` field at every `ckpt_every_steps`.
- The final student model is saved according to the `checkpoint` section (e.g., `model_save_format: safetensors`, consolidated weights if `save_consolidated: true`).

Load the distilled model:

```python
import nemo_automodel as am
student = am.NeMoAutoModelForCausalLM.from_pretrained("checkpoints/final")
print(student("Translate to French: I love coding!").text)
```