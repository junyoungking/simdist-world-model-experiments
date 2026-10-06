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
repo="$portfolio_root"
root="$artifact_root/module_combo_v1/mass8"
old="$artifact_root/extended_adaptation_v1/oracle/collections"
python_bin="${SIMDIST_PYTHON:-python}"
model="$checkpoint_root/models/go2_2026-02-12"
model90="$artifact_root/extended_adaptation_v1/training/90k/checkpoint"
model30p="$artifact_root/followup_v1/oracle_snapshots/update_50/checkpoint"
mkdir -p "$root/policy" "$root/evaluations" "$root/logs"

train_paths=()
for seed in {33801..33805}; do train_paths+=("$old/seed_$seed/rollout.h5"); done
heldout="$old/seed_33806/rollout.h5"
for stage in m0 90k; do
  if [[ "$stage" == m0 ]]; then previous="$model"; else previous="$model90"; fi
  out="$root/policy/$stage"
  if [[ ! -f "$out/summary.json" ]]; then
    "$python_bin" -B "$portfolio_root/experiments/module_combo_v1/train_policy_50.py" \
      --previous "$previous" --train-rollouts "${train_paths[@]}" \
      --heldout-rollout "$heldout" --out "$out" > "$root/logs/train_policy_$stage.log" 2>&1
  fi
  echo "TRAINED mass8 $stage"
done

seeds=()
for ((seed=23501; seed<=23600; seed++)); do seeds+=("$seed"); done
for name in m0p_mppi d30p_mppi m0p_base d90p_mppi; do
  case "$name" in
    m0p_mppi) controller=mppi; checkpoint="$root/policy/m0/checkpoint" ;;
    d30p_mppi) controller=mppi; checkpoint="$model30p" ;;
    m0p_base) controller=base; checkpoint="$root/policy/m0/checkpoint" ;;
    d90p_mppi) controller=mppi; checkpoint="$root/policy/90k/checkpoint" ;;
  esac
  out="$root/evaluations/$name"
  "$python_bin" -B "$portfolio_root/experiments/followup_v1/eval_gap_batch.py" \
    --gap mass8 --controller "$controller" --forward-vel 1.0 \
    --seeds "${seeds[@]}" --checkpoint "$checkpoint" --out-root "$out" \
    > "$root/logs/eval_$name.log" 2>&1
  echo "EVALUATED mass8 $name"
done
