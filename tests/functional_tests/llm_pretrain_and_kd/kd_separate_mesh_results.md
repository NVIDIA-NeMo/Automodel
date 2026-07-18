# Separate-Mesh Knowledge Distillation Validation

The original table in this file was captured before `KDLoss` was corrected
from teacher-to-student soft cross-entropy to zero-based forward KL. Those
absolute values are intentionally removed because they no longer represent the
metric reported by the code.

Current BF16 loss, gradient, restart, and TP/PP/EP parity evidence is recorded
in `kd_sep_mesh_cw_dfw_results.md`. The small four-rank CPU bridge smoke remains
available as:

```bash
torchrun --standalone --nproc-per-node=4 \
  tests/functional_tests/llm_pretrain_and_kd/run_kd_separate_meshes.py
```

The corresponding user-facing four-GPU configurations are:

- `examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_tp2.yaml`
- `examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_cp2.yaml`
- `examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_pp2.yaml`
