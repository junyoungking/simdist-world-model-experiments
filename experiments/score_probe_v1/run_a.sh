#!/usr/bin/env bash
set -euo pipefail

# Portable roots; preserve the original seeds, budgets and flags.
portfolio_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SIMDIST_ROOT:?Set SIMDIST_ROOT to the pinned external SimDist checkout}"
artifact_root="${SIMDIST_ARTIFACT_ROOT:-$portfolio_root/results}"
checkpoint_root="${SIMDIST_CHECKPOINT_ROOT:-$SIMDIST_ROOT/checkpoints}"
export SIMDIST_ARTIFACT_ROOT="$artifact_root"
export PYTHONPATH="$portfolio_root:$SIMDIST_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$portfolio_root"
root="$artifact_root/score_probe_v1"
python=${SIMDIST_PYTHON:-python}
mkdir -p "$root/logs" "$root/evaluations"

for label in m0 cp10000 cp30000 base; do
  echo "EVALUATE $label seeds 23521..23600"
  case "$label" in
    m0) controller=mppi; checkpoint=() ;;
    cp10000) controller=mppi; checkpoint=(--checkpoint "$artifact_root/adaptation_curve_v1/training/stage_10000/checkpoint") ;;
    cp30000) controller=mppi; checkpoint=(--checkpoint "$artifact_root/adaptation_curve_v1/training/stage_30000/checkpoint") ;;
    base) controller=base; checkpoint=() ;;
  esac
  PYTHONUNBUFFERED=1 timeout 8100 "$python" -B "$portfolio_root/experiments/score_probe_v1/eval_probe_batch.py" \
    --controller "$controller" --mass 8 --forward-vel 1.0 \
    --seeds {23521..23600} --probe-every 5 \
    --out-root "$root/evaluations/$label" "${checkpoint[@]}" \
    > "$root/logs/eval_${label}.log" 2>&1
done
