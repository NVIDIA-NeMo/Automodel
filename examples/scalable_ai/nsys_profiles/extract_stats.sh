#!/bin/bash
# Export CSV summaries from the nsys reports produced by regenerate_profiles.sh, so the stage ladder can
# be read without nsys-ui.  Writes to nsys_profiles/stats/, one CSV per report per profile.
#
#   ./extract_stats.sh                          # the stage ladder (stage1..stage5)
#   ./extract_stats.sh all                      # every .nsys-rep in the folder
#   ./extract_stats.sh stage3_groupedgemm ...   # named profiles
#
# FORCE=1 re-exports profiles whose CSVs already exist.  nsys builds a .sqlite beside each report on the
# first run, which is the slow part; later runs reuse it unless FORCE=1 passes --force-export.
set +e   # keep going if one report fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/stats"
mkdir -p "$OUT"

# nvtx_pushpop_sum and nvtx_gpu_proj_sum are the reason regenerate_profiles.sh passes `--nvtx true`:
# they resolve to module names (DeepseekV4Attention, DeepseekV4Indexer, MoE, GroupedExperts) instead of
# raw kernel names.  cuda_api_sum carries the kernel *launch* counts, which is where the per-expert loop
# and the grouped GEMM differ most visibly.
#
# nvtx_kern_sum is the one that makes per-module numbers safe to quote.  nvtx_gpu_proj_sum's `Proj *`
# columns are the *span* of a range on the GPU timeline, gaps included, so a module whose kernels are
# unchanged but whose launches spread out looks slower.  nvtx_kern_sum sums kernel execution time inside
# each range instead, which is the number to use when a stage delta is small enough that a span could
# explain it.  See README.md "What the ladder measures" for where this bit.
REPORTS=(nvtx_pushpop_sum nvtx_gpu_proj_sum nvtx_kern_sum cuda_gpu_kern_sum cuda_gpu_sum cuda_api_sum)
STAGES=(stage1_hf_ootb stage2_am_expertloop stage3_groupedgemm stage4_deepep stage5_tilelang)

# nsys wants `--report X --report Y`, so each flag and value must be its own argv entry.
REPORT_ARGS=()
for r in "${REPORTS[@]}"; do REPORT_ARGS+=(--report "$r"); done

case "${1:-}" in
  "")    profiles=("${STAGES[@]}") ;;
  all)   profiles=(); for f in "$SCRIPT_DIR"/*.nsys-rep; do [ -e "$f" ] && profiles+=("$(basename "$f" .nsys-rep)"); done ;;
  *)     profiles=("$@") ;;
esac

if [ ${#profiles[@]} -eq 0 ]; then
  echo "No profiles to process. Run regenerate_profiles.sh first."
  exit 1
fi

for name in "${profiles[@]}"; do
  report="$SCRIPT_DIR/$name.nsys-rep"
  if [ ! -s "$report" ]; then
    echo "=== $name -- no report, skipping (run regenerate_profiles.sh) ==="
    continue
  fi
  # One CSV per report is written as <name>_<report>.csv.  Require *every* report's CSV before calling a
  # profile cached, not just the first: otherwise adding a report to REPORTS silently serves the old set
  # forever, since the marker file would already exist.
  missing=0
  for r in "${REPORTS[@]}"; do [ -s "$OUT/${name}_${r}.csv" ] || missing=1; done
  if [[ "$missing" == "0" && "${FORCE:-0}" != "1" ]]; then
    echo "=== $name -- cached, skipping (FORCE=1 to re-export) ==="
    continue
  fi
  echo "=== $name ==="
  nsys stats --format csv --output "$OUT" \
    ${FORCE:+--force-export=true} \
    "${REPORT_ARGS[@]}" "$report" 2>&1 | grep -vE "^Processing|^Exporting|^\s*$"
done

echo
echo "=== CSVs in $OUT ==="
ls -1 "$OUT"/*.csv 2>/dev/null | sed "s|$OUT/||" || echo "(none written)"
