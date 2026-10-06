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
out_base="$artifact_root/followup_v1/chain/collections/$stage"
python_bin="${SIMDIST_PYTHON:-python}"
if [[ "$stage" == "10k" ]]; then
  checkpoint="$checkpoint_root/models/go2_2026-02-12"
  first_seed=36101
elif [[ "$stage" == "30k" ]]; then
  checkpoint="$artifact_root/followup_v1/chain/training/10k/checkpoint"
  first_seed=36201
else
  echo "expected 10k or 30k" >&2
  exit 2
fi
mkdir -p "$out_base"
for offset in 0 1 2 3 4 5; do
  seed=$((first_seed + offset))
  if [[ "$stage" == "10k" ]]; then
    steps=1667
    if [[ "$offset" -ge 4 ]]; then steps=1666; fi
  else
    steps=3333
    if [[ "$offset" -lt 2 ]]; then steps=3334; fi
  fi
  out="$out_base/seed_$seed"
  if [[ -f "$out/summary.json" ]]; then
    echo "ALREADY_DONE $stage $seed"
    continue
  fi
  "$python_bin" -B "$portfolio_root/experiments/adaptation_curve_v1/collect.py" \
    --checkpoint "$checkpoint" --seed "$seed" --steps "$steps" --out "$out" \
    > "$out_base/seed_$seed.log" 2>&1
  echo "DONE $stage $seed"
done
