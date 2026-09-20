#!/bin/bash
# Export CSV summaries from the nsys reports produced by regenerate_profiles.sh, so the stage ladder can
# be read without nsys-ui.  Writes to nsys_profiles/stats/, one CSV per report per profile.
#
#   ./extract_stats.sh                          # the stage ladder (stage1..stage5)
#   ./extract_stats.sh all                      # every .nsys-rep in the folder
#   ./extract_stats.sh stage3_groupedgemm ...   # named profiles
#
# Only the reports whose CSV is missing are requested, so adding an entry to REPORTS costs one report per
# profile rather than a full re-export.  FORCE=1 re-requests every report; FORCE_EXPORT=1 additionally
# rebuilds the .sqlite nsys keeps beside each .nsys-rep, which is the slow part and is only needed when a
# profile was regenerated in place.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/stats"
GEN="$SCRIPT_DIR/regenerate_profiles.sh"
mkdir -p "$OUT"
shopt -s nullglob   # an unmatched *.nsys-rep glob must expand to nothing, not to itself

# nvtx_pushpop_sum and nvtx_gpu_proj_sum are the reason regenerate_profiles.sh passes `--nvtx true`:
# they resolve to module names (DeepseekV4Attention, DeepseekV4Indexer, MoE, GroupedExperts) instead of
# raw kernel names.  cuda_api_sum carries the kernel *launch* counts, which is where the per-expert loop
# and the grouped GEMM differ most visibly.  nvtx_kern_sum sums kernel time inside each range, unlike
# nvtx_gpu_proj_sum's timeline span -- see README.md "What the ladder measures" for when that matters.
REPORTS=(nvtx_pushpop_sum nvtx_gpu_proj_sum nvtx_kern_sum cuda_gpu_kern_sum cuda_gpu_sum cuda_api_sum)

# The ladder's membership lives in regenerate_profiles.sh, which is what actually produces the reports.
# Read it back rather than restating it here, so a renamed or added stage cannot go missing from the stats.
STAGES=()
# Stages 4-5 sit indented inside the RUN_HOPPER block, so the match cannot be anchored at column 0.
while IFS= read -r s; do STAGES+=("$s"); done < <(grep -oE '^[[:space:]]*e2e +stage[A-Za-z0-9_]*' "$GEN" | awk '{print $NF}')
if [ ${#STAGES[@]} -eq 0 ]; then
  echo "No 'e2e stage...' lines found in $GEN -- has the ladder been renamed?"
  exit 1
fi

case "${1:-}" in
  "")    profiles=("${STAGES[@]}") ;;
  all)   profiles=(); for f in "$SCRIPT_DIR"/*.nsys-rep; do f="${f##*/}"; profiles+=("${f%.nsys-rep}"); done ;;
  *)     profiles=("$@") ;;
esac

if [ ${#profiles[@]} -eq 0 ]; then
  echo "No profiles to process. Run regenerate_profiles.sh first."
  exit 1
fi

failed=0
for name in "${profiles[@]}"; do
  report="$SCRIPT_DIR/$name.nsys-rep"
  if [ ! -s "$report" ]; then
    echo "=== $name -- no report, skipping (run regenerate_profiles.sh) ==="
    continue
  fi
  # One CSV per report is written as <name>_<report>.csv.  Ask only for the ones that are not there yet:
  # a presence check per report is what lets REPORTS grow without either serving a stale set or redoing
  # the reports that are already current.
  want=()
  for r in "${REPORTS[@]}"; do
    if [[ "${FORCE:-0}" == "1" || ! -s "$OUT/${name}_${r}.csv" ]]; then want+=("$r"); fi
  done
  if [ ${#want[@]} -eq 0 ]; then
    echo "=== $name -- cached, skipping (FORCE=1 to re-request) ==="
    continue
  fi
  echo "=== $name -- ${want[*]} ==="
  report_args=()
  for r in "${want[@]}"; do report_args+=(--report "$r"); done
  # Run from inside $OUT with `--output .`.  nsys treats --output as a filename *prefix*, not a
  # directory, so `--output "$OUT"` writes "<OUT>_<report>.csv" beside the profiles -- one name shared by
  # every stage, so the first wins and the rest are SKIPPED.  A directory of "." makes it name each file
  # after the report it read, which is what gives <profile>_<report>.csv.
  log="$(cd "$OUT" && nsys stats --format csv --output . \
           ${FORCE_EXPORT:+--force-export=true} \
           "${report_args[@]}" "$report" 2>&1)"
  status=$?

  # nsys exits 0 and prints its usual NOTICE even when a requested report writes nothing, so verify the
  # artifacts rather than the exit status.  On failure print the unfiltered log: the filter below is a
  # convenience for the normal path and must never be the reason a diagnostic is lost.
  absent=()
  for r in "${want[@]}"; do [ -s "$OUT/${name}_${r}.csv" ] || absent+=("$r"); done
  if [ ${#absent[@]} -ne 0 ]; then
    failed=1
    echo "    FAILED: nsys exited $status but wrote no CSV for: ${absent[*]}"
    echo "    expected: $OUT/${name}_<report>.csv"
    printf '%s\n' "$log" | sed 's/^/    | /'
  else
    printf '%s\n' "$log" | grep -vE "^Processing|^Exporting|^\s*$"
  fi
done

echo
echo "=== CSVs in $OUT ==="
csvs=("$OUT"/*.csv)
if [ ${#csvs[@]} -eq 0 ]; then echo "(none written)"; else printf '%s\n' "${csvs[@]##*/}"; fi

if [ "$failed" -ne 0 ]; then
  echo
  echo "One or more reports produced no CSV -- see the FAILED lines above."
  exit 1
fi
