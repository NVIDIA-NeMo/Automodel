# Separate-Mesh Knowledge Distillation Results

This document records the losses from the 4-GPU validation of separate student
and teacher device meshes. All runs used the tiny Llama functional-test fixture,
sequence length 8, `kd_ratio=0.5`, and temperature 1.0.

The student used ranks `[0, 1]` with `dp=2`, `tp=1`, `cp=1`, and `pp=1` in
every run. The teacher used ranks `[2, 3]`; its layout varied across tensor,
context, and pipeline parallelism as shown below.

## Losses

| Teacher layout | Step | Epoch | Total loss | CE loss | KD loss | Gradient norm |
|---|---:|---:|---:|---:|---:|---:|
| TP2 (`dp=1,tp=2,cp=1,pp=1`) | 0 | 0 | 3.466952323913574 | 3.4752421379089355 | 3.458662509918213 | 1.5543898344039917 |
| TP2 (`dp=1,tp=2,cp=1,pp=1`) | 1 | 0 | 3.4720077514648438 | 3.484994888305664 | 3.4590208530426025 | 1.6541537046432495 |
| TP2 (`dp=1,tp=2,cp=1,pp=1`) | 2 | 0 | 3.4363951683044434 | 3.413518190383911 | 3.4592719078063965 | 1.6042704582214355 |
| TP2 (`dp=1,tp=2,cp=1,pp=1`) | 3 | 0 | 3.436145305633545 | 3.413287878036499 | 3.4590024948120117 | 1.5892447233200073 |
| TP2 (`dp=1,tp=2,cp=1,pp=1`) | 4 | 0 | 3.440131187438965 | 3.420288324356079 | 3.4599742889404297 | 1.7026044130325317 |
| TP2 (`dp=1,tp=2,cp=1,pp=1`) | 5 | 0 | 3.4552316665649414 | 3.4500961303710938 | 3.460367202758789 | 1.639026165008545 |
| TP2 (`dp=1,tp=2,cp=1,pp=1`) | 6 | 0 | 3.470043659210205 | 3.480128049850464 | 3.4599595069885254 | 1.6409136056900024 |
| TP2 (`dp=1,tp=2,cp=1,pp=1`) | 7 | 0 | 3.4858603477478027 | 3.5123836994171143 | 3.459336757659912 | 1.463856816291809 |
| TP2 (`dp=1,tp=2,cp=1,pp=1`) | 8 | 1 | 3.4410319328308105 | 3.4232616424560547 | 3.4588022232055664 | 1.538885474205017 |
| TP2 (`dp=1,tp=2,cp=1,pp=1`) | 9 | 1 | 3.442882537841797 | 3.4264957904815674 | 3.4592690467834473 | 1.6266632080078125 |
| CP2 (`dp=1,tp=1,cp=2,pp=1`) | 0 | 0 | 3.4669547080993652 | 3.4752421379089355 | 3.458667755126953 | 1.5543898344039917 |
| PP2 (`dp=1,tp=1,cp=1,pp=2`) | 0 | 0 | 3.471512794494629 | 3.4841909408569336 | 3.458834648132324 | 1.452264428138733 |

Every value is finite. For every row, the reported total loss agrees with
`0.5 * CE loss + 0.5 * KD loss` within `1e-5`.

## Example configurations

The corresponding user-facing four-GPU configurations are:

- `examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_tp2.yaml`
- `examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_cp2.yaml`
- `examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_pp2.yaml`

## Commands

The TP2 run used the fixture defaults:

```bash
torchrun --standalone --nproc-per-node=4 \
  nemo_automodel/recipes/llm/kd.py \
  -c tests/functional_tests/llm_pretrain_and_kd/kd_separate_mesh.yaml \
  --step_scheduler.max_steps 10 \
  --step_scheduler.num_epochs 2
```

The CP2 run changed the teacher layout and recorded one step:

```bash
torchrun --standalone --nproc-per-node=4 \
  nemo_automodel/recipes/llm/kd.py \
  -c tests/functional_tests/llm_pretrain_and_kd/kd_separate_mesh.yaml \
  --teacher_distributed.tp_size 1 \
  --teacher_distributed.cp_size 2 \
  --step_scheduler.max_steps 1
```

The PP2 run changed the teacher layout, used two student samples per local
batch, and recorded one step:

```bash
torchrun --standalone --nproc-per-node=4 \
  nemo_automodel/recipes/llm/kd.py \
  -c tests/functional_tests/llm_pretrain_and_kd/kd_separate_mesh.yaml \
  --teacher_distributed.tp_size 1 \
  --teacher_distributed.pp_size 2 \
  --step_scheduler.local_batch_size 2 \
  --step_scheduler.global_batch_size 4 \
  --step_scheduler.max_steps 1
```
