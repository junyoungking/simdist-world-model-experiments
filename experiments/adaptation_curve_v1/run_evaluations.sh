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
root="$artifact_root/adaptation_curve_v1"
python=${SIMDIST_PYTHON:-python}
mkdir -p "$root/logs" "$root/evaluations"

# The first ten baseline episodes are the identical v2 runs in baseline_v2/runs.
for specification in "mppi 8 baseline_target" "base 8 baseline_target" "expert 8 baseline_target" "mppi 0 baseline_source"; do
  read -r controller mass label <<< "$specification"
  echo "EVALUATE baseline $label $controller mass=$mass seeds=23511..23520"
  PYTHONUNBUFFERED=1 timeout 1800 "$python" -B "$portfolio_root/experiments/adaptation_curve_v1/eval_batch.py" \
    --controller "$controller" --mass "$mass" --forward-vel 1.0 \
    --seeds {23511..23520} --out-root "$root/evaluations/$label" \
    > "$root/logs/eval_${label}_${controller}.log" 2>&1
done

for stage in 1000 3000 10000 30000; do
  echo "EVALUATE stage=$stage mass=8 seeds=23501..23520"
  PYTHONUNBUFFERED=1 timeout 2400 "$python" -B "$portfolio_root/experiments/adaptation_curve_v1/eval_batch.py" \
    --controller mppi --mass 8 --forward-vel 1.0 --seeds {23501..23520} \
    --checkpoint "$root/training/stage_${stage}/checkpoint" \
    --out-root "$root/evaluations/stage_${stage}" \
    > "$root/logs/eval_stage_${stage}.log" 2>&1
done

"$python" -B "$portfolio_root/analysis/adaptation_curve_v1/aggregate.py" > "$root/logs/aggregate.log"
