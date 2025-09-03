# Knowledge Distillation with NeMo-AutoModel

This guide walks through fine-tuning a **student** LLM with the help of a
larger **teacher** model using the new `knowledge_distillation` recipe.

> **TL;DR**  
> ```bash
> automodel knowledge_distillation llm \
>     -c examples/llm/llama_3_2_1b_kd.yaml \
>     --nproc-per-node 8
> ```

---

## 1. What is Knowledge Distillation?

Knowledge distillation (KD) transfers the *dark knowledge* of a high-capacity
teacher model to a smaller student by minimizing the divergence between their
predicted distributions.  The student learns from both the ground-truth labels
(Cross-Entropy loss, **CE**) and the soft targets of the teacher (Kullback-Leibler
loss, **KD**):

\[ \mathcal{L}= (1-\alpha)\;\mathcal{L}_{\text{CE}} + \alpha\;T^{2}\;\mathcal{L}_{\text{KD}} \]

where \(\alpha\) is `kd_ratio` and \(T\) is `temperature`.

---

## 2. Prepare the YAML config

A ready-to-use example is provided at
`examples/llm/llama_3_2_1b_kd.yaml`.  Important sections:

* `model` – the student to be fine-tuned (1 B parameters in the example)
* `teacher_model` – a larger frozen model used for supervision (7 B)
* `kd_ratio` – blend between CE and KD loss
* `temperature` – softens probability distributions before KL-divergence
* `peft` – **optional** LoRA config (commented). Uncomment to train only a
  handful of parameters.

Feel free to tweak these values as required.

---

## 3. Launch training

### Single-GPU quick run

```bash
# Runs on a single device of the current host
PYTORCH_ENABLE_MPS_FALLBACK=1 \
automodel knowledge_distillation llm -c examples/llm/llama_3_2_1b_kd.yaml
```

### Multi-GPU (single node)

```bash
# Leverage all GPUs on the local machine
torchrun --nproc-per-node $(nvidia-smi -L | wc -l) \
    nemo_automodel/recipes/llm/knowledge_distillation.py \
    -c examples/llm/llama_3_2_1b_kd.yaml
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

• Checkpoints are written under `checkpoints/` at every `ckpt_every_steps`.  
• The final student model is saved in both "native" and consolidated
`safetensors` formats for hassle-free deployment.

Load the distilled model:

```python
import nemo_automodel as am
student = am.NeMoAutoModelForCausalLM.from_pretrained("checkpoints/final")
print(student("Translate to French: I love coding!").text)
```

---

Happy distilling! 🎉 