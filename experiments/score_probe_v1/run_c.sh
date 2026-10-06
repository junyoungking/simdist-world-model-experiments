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
mkdir -p "$root/logs" "$root/validation_collections"

for label in m0 cp10000; do
  if [[ "$label" == m0 ]]; then
    checkpoint="$checkpoint_root/models/go2_2026-02-12"
    step=(--step 367004)
  else
    checkpoint="$artifact_root/adaptation_curve_v1/training/stage_10000/checkpoint"
    step=()
  fi
  for seed in {35001..35010}; do
    out="$root/validation_collections/$label/seed_$seed"
    if [[ -f "$out/summary.json" ]]; then
      echo "DONE validation $label $seed"
      continue
    fi
    echo "COLLECT validation $label $seed"
    PYTHONUNBUFFERED=1 timeout 600 "$python" -B \
      "$portfolio_root/experiments/adaptation_curve_v1/collect.py" --checkpoint "$checkpoint" \
      "${step[@]}" --seed "$seed" --steps 500 --out "$out" \
      > "$root/logs/collect_${label}_${seed}.log" 2>&1
  done
done
