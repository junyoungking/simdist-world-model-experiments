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
base="$artifact_root/followup_v1/chain"
python_bin="${SIMDIST_PYTHON:-python}"
if [[ "$stage" == "10k" ]]; then
  previous="$checkpoint_root/models/go2_2026-02-12"
  train_paths=()
  for seed in {36101..36105}; do
    train_paths+=("$base/collections/10k/seed_$seed/rollout.h5")
  done
  heldout_paths=("$base/collections/10k/seed_36106/rollout.h5")
  latest="${heldout_paths[0]}"
  training_seed=36501
elif [[ "$stage" == "30k" ]]; then
  previous="$base/training/10k/checkpoint"
  train_paths=()
  for seed in {36101..36105}; do
    train_paths+=("$base/collections/10k/seed_$seed/rollout.h5")
  done
  for seed in {36201..36205}; do
    train_paths+=("$base/collections/30k/seed_$seed/rollout.h5")
  done
  heldout_paths=("$base/collections/10k/seed_36106/rollout.h5"
                 "$base/collections/30k/seed_36206/rollout.h5")
  latest="${heldout_paths[1]}"
  training_seed=36502
else
  echo "expected 10k or 30k" >&2
  exit 2
fi
"$python_bin" -B "$portfolio_root/experiments/extended_adaptation_v1/train_seed_split.py" \
  --previous "$previous" --seed "$training_seed" \
  --train-rollouts "${train_paths[@]}" \
  --heldout-rollouts "${heldout_paths[@]}" \
  --latest-heldout "$latest" --out "$base/training/$stage"
