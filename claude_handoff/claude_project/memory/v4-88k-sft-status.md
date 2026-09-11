---
name: v4-88k-sft-status
description: "State of the Qwen3.6-35B-A3B v4_88k SFT setup on 2xB300 as of 2026-09-11 — rung results, decisions, perf numbers, GPU0 hardware fault, next steps"
metadata: 
  node_type: memory
  type: project
  originSessionId: dc22bf29-146a-4a63-8910-37a0e87d9eb4
  modified: 2026-09-11T09:14:11.465Z
---

Full narrative + reproduction steps: /workspace/CLAUDE_HANDOFF.md. Env details: [[runpod-b300-env-setup]]. Preferences: [[user-working-preferences]].

**Status 2026-09-11 ~09:15: BLOCKED — GPU 0 (PCI 9A:00.0) hardware fault.** nvidia-smi: "GPU Recovery Action: Reset", Channel/TPC Repair Pending, 5 uncorrectable retired pages pending, all NVLinks inactive. DDP probe died with "CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error" in checkpoint load. Needs RunPod reset/new pod. Re-verify health (ECC, `nvidia-smi nvlink -s`, DeepEP test_intranode --num-processes 2) before any run.

**Runbook rungs (recipe examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_v4_88k_ep8.yaml, run with --nproc-per-node 2 --distributed.ep_size 2):** 1 kernels ok; 2 masking ok after fix; 3 prefilter 87,279/87,552 kept (99.69%); 4 config ok; 5 proxy ok; 6 smoke ok. Rung 7 (200 steps) NOT started. Short runs (<51 steps) need --lr_scheduler.lr_warmup_steps 1.

**User decisions:** mask empty think block (model trained for enable_thinking=False); optimizer torchao AdamW8bit (+ scheduler.py fix for tensor lr); keep attn: te (no attention code changes); no custom samplers; no FP8 without asking.

**Perf (FSDP2+EP2, TE 2.18 fused attn, AC on, real data):** lbs1 ~79 s/step (138.6 GiB peak); lbs2 ~64 s/step, 4.8–7k tps (166 GiB); lbs4 fits worst case (256 longest rows, ~192 GiB) but speed not measured; lbs8 OOM. Padding with random batching: lbs2 32%, lbs4 49%, lbs8 58%. grad_norm 3–7 (clip 1.0) in first 5 steps — watch in rung 7.
**DDP (examples/.../qwen3_6_35b_v4_88k_ddp2.yaml, strategy ddp, ep1, dispatcher torch/experts torch_mm):** each rank loads full model on CPU, ~13 min, single-threaded (OMP_NUM_THREADS doesn't help); fit unknown (crashed on the GPU fault). FSDP2 dp_replicate_size=2 on 2 GPUs is rejected by mesh_utils.

**Uncommitted repo changes (branch trungvd-zenai/feat/qwen3-6-v4-88k-sft):** v4_88k.py + check_masking (masking), recipe YAML (AdamW8bit), scheduler.py + test (tensor lr), RUNBOOK (notes), new ddp2 YAML. Patch backup in /workspace/claude_handoff/.

**Why:** multi-day run planned; resume from here rather than re-deriving.
**How to apply:** after a new/reset pod, check GPU health first, then re-validate rung 1 + a 5-step smoke before rung 7.
