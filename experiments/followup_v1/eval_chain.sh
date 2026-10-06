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
repo_root="$portfolio_root"
base="$artifact_root/followup_v1/chain"
seeds=()
for ((seed=23501; seed<=23600; seed++)); do seeds+=("$seed"); done
"${SIMDIST_PYTHON:-python}" -B \
  "$portfolio_root/experiments/adaptation_curve_v1/eval_batch.py" --controller mppi \
  --mass 8 --forward-vel 1.0 --seeds "${seeds[@]}" \
  --checkpoint "$base/training/30k/checkpoint" \
  --out-root "$base/evaluation"
