#!/usr/bin/env bash
# Restore Claude Code memory + session transcript on a new/reset pod.
# Prerequisite: /workspace (this folder) carried over; Claude Code installed and logged in.
set -eu
H=/workspace/claude_handoff
DST=/root/.claude/projects/-workspace     # project key for cwd=/workspace
mkdir -p "$DST"
cp -an "$H/claude_project/." "$DST/"      # -n: never clobber a newer local file
[ -f "$H/claude_home/settings.json" ] && [ ! -f /root/.claude/settings.json ] && cp -a "$H/claude_home/settings.json" /root/.claude/
echo "Restored memory: $(ls "$DST/memory" | tr '\n' ' ')"
echo
echo "Next:"
echo "  1. Read /workspace/CLAUDE_HANDOFF.md (start with section 0: GPU health checks)."
echo "  2. Rebuild the env if the venv/toolkit are missing: section 3 of the handoff."
echo "  3. Resume this exact session:   cd /workspace && claude --resume dc22bf29-146a-4a63-8910-37a0e87d9eb4"
echo "     or start fresh:              cd /workspace && claude   (memory loads automatically)"
echo "  4. Repo changes if the checkout is fresh:"
echo "       cd /workspace/Automodel && git checkout trungvd-zenai/feat/qwen3-6-v4-88k-sft && git apply $H/uncommitted.patch && cp -r $H/untracked/. ."
