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
base="$(pwd)/$artifact_root/extended_adaptation_v1/oracle/collections"
mkdir -p "$base"
python_bin="${SIMDIST_PYTHON:-python}"
for offset in 0 1 2 3 4 5; do
  seed=$((33801 + offset))
  steps=1667
  if [[ "$offset" -ge 4 ]]; then
    steps=1666
  fi
  out="$base/seed_$seed"
  if [[ -f "$out/summary.json" ]]; then
    echo "EXPERT_COLLECTION_ALREADY_DONE $seed"
    continue
  fi
  "$python_bin" -B "$portfolio_root/experiments/extended_adaptation_v1/collect_expert.py" \
    --seed "$seed" --steps "$steps" --out "$out" > "$base/seed_$seed.log" 2>&1
  echo "EXPERT_COLLECTION_DONE $seed"
done
