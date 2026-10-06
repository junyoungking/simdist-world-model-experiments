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
base="$artifact_root/extended_adaptation_v1"
old="$artifact_root/adaptation_curve_v1/collections"
independent="$artifact_root/score_probe_v1/validation_collections/cp10000/seed_35001/rollout.h5"
python_bin="${SIMDIST_PYTHON:-python}"
train_paths=(
  "$old/stage_1000/rollout.h5" "$old/stage_3000/rollout.h5"
  "$old/stage_10000/rollout.h5" "$old/stage_30000/rollout.h5"
)
heldout_paths=("$independent")
for seed in 33601 33602 33603 33604 33605; do
  train_paths+=("$base/collections/60k/seed_$seed/rollout.h5")
done
heldout_paths+=("$base/collections/60k/seed_33606/rollout.h5")
if [[ "$stage" == "60k" ]]; then
  previous="$base/training/30k_c/checkpoint"
  latest="$base/collections/60k/seed_33606/rollout.h5"
elif [[ "$stage" == "90k" ]]; then
  previous="$base/training/60k/checkpoint"
  for seed in 33701 33702 33703 33704 33705; do
    train_paths+=("$base/collections/90k/seed_$seed/rollout.h5")
  done
  heldout_paths+=("$base/collections/90k/seed_33706/rollout.h5")
  latest="$base/collections/90k/seed_33706/rollout.h5"
else
  echo "expected 60k or 90k" >&2
  exit 2
fi
"$python_bin" -B "$portfolio_root/experiments/extended_adaptation_v1/train_seed_split.py" \
  --previous "$previous" --train-rollouts "${train_paths[@]}" \
  --heldout-rollouts "${heldout_paths[@]}" --latest-heldout "$latest" \
  --out "$base/training/$stage"
