#!/usr/bin/env bash
# Snapshot everything a new Claude Code session needs into /workspace/claude_handoff/.
# Re-run right before shutting the pod down to capture the latest transcript.
set -eu
H=/workspace/claude_handoff
SRC_PROJ=/root/.claude/projects/-workspace          # memory + session transcripts (cwd=/workspace)
SCRATCH=/tmp/claude-0/-workspace/dc22bf29-146a-4a63-8910-37a0e87d9eb4/scratchpad
mkdir -p "$H/claude_project" "$H/scratch" "$H/claude_home"

# 1) Claude project dir: memory/ + <session>.jsonl transcript(s) + per-session dirs.
cp -a "$SRC_PROJ/." "$H/claude_project/"

# 2) Non-secret Claude home files. .credentials.json is deliberately NOT copied (auth secret).
for f in settings.json history.jsonl; do
  [ -f "/root/.claude/$f" ] && cp -a "/root/.claude/$f" "$H/claude_home/"
done

# 3) Scratchpad probes (small files only; skip re-downloadable wheels and test checkpoints).
if [ -d "$SCRATCH" ]; then
  find "$SCRATCH" -maxdepth 1 -type f \( -name '*.py' -o -name '*.txt' \) -exec cp -a {} "$H/scratch/" \;
  # DeepEP clone at the uv.lock rev (42144303): its tests/test_intranode.py is the GPU/NVLink health check.
  [ -d "$SCRATCH/DeepEP" ] && rm -rf "$H/scratch/DeepEP" && cp -a "$SCRATCH/DeepEP" "$H/scratch/DeepEP"
fi

# 4) Uncommitted repo work: tracked diff + untracked files, plus branch/commit for context.
cd /workspace/Automodel
git diff > "$H/uncommitted.patch"
rm -rf "$H/untracked"; mkdir -p "$H/untracked"
git ls-files --others --exclude-standard -z | xargs -0 -r -I{} cp --parents {} "$H/untracked/"
{ echo "branch: $(git branch --show-current)"; echo "base commit: $(git rev-parse HEAD)"; git status --short; } > "$H/git_state.txt"

# 5) Current package versions (the TE/cuDNN/nvidia-ml-py installs are outside uv.lock).
UV_CACHE_DIR=/workspace/.uv_cache uv pip freeze --python /workspace/Automodel/.venv/bin/python > "$H/venv_freeze_current.txt" 2>/dev/null || true

date '+backup taken %F %T' > "$H/BACKUP_TIMESTAMP"
echo "backup ok -> $H"; du -sh "$H"
