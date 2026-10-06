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
stage="$1"
repo_root="$portfolio_root"
out_base="$artifact_root/extended_adaptation_v1/collections/$stage"
python_bin="${SIMDIST_PYTHON:-python}"
if [[ "$stage" == "60k" ]]; then
  checkpoint="$artifact_root/extended_adaptation_v1/training/30k_c/checkpoint"
  first_seed=33601
elif [[ "$stage" == "90k" ]]; then
  checkpoint="$artifact_root/extended_adaptation_v1/training/60k/checkpoint"
  first_seed=33701
else
  echo "expected 60k or 90k" >&2
  exit 2
fi
mkdir -p "$out_base"
for offset in 0 1 2 3 4 5; do
  seed=$((first_seed + offset))
  out="$out_base/seed_$seed"
  if [[ -f "$out/summary.json" ]]; then
    echo "COLLECTION_ALREADY_DONE $stage $seed"
    continue
  fi
  "$python_bin" -B "$portfolio_root/experiments/adaptation_curve_v1/collect.py" \
    --checkpoint "$checkpoint" --seed "$seed" --steps 5000 --out "$out" \
    > "$out_base/seed_$seed.log" 2>&1
  echo "COLLECTION_DONE $stage $seed"
done
