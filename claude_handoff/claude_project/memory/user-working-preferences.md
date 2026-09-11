---
name: user-working-preferences
description: "How this user wants changes made on the Automodel/v4_88k work — minimal codebase edits, predefined components, learnability first, stop-and-ask on semantics"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: dc22bf29-146a-4a63-8910-37a0e87d9eb4
  modified: 2026-09-11T09:13:55.919Z
---

- **Minimal changes to the codebase's attention mechanism.** When I patched `components/attention/utils.py` (right-padding → flash SDPA) the user had it reverted and preferred fixing the environment instead (TE 2.18 + cuDNN 9.26 + CUDNN_HOME).
  **Why:** they want upstream behavior preserved; performance should come from packages/config, not model/attention patches.
  **How to apply:** exhaust package/config/env options before touching model or attention code; ask before editing shared components.
- **Prefer predefined samplers/components over new code.** A new `StepWindowLengthSortedSampler` I wrote was reverted on request ("I prefer using the predefined samplers").
  **How to apply:** propose existing repo components first; if none fit, explain why and ask before writing new ones.
- **Learnability first.** User explicitly asked not to fall into traps that hurt learnability (e.g. length-grouped batching that changes step composition, FP8 numerics). Flag numerics-changing options; never enable them silently.
- **Masking decision is theirs.** For v4_88k they chose to mask the empty `<think>\n\n</think>\n\n` block and wanted it noted (it is, in the runbook). See [[v4-88k-sft-status]].
- Wants maximal performance before committing to a long run, but measured and one knob at a time.
- Gives terse instructions ("stop", "ETA?", "how is it going?"): answer with current state + numbers first.
- Do not commit unless asked (nothing committed as of 2026-09-11).
