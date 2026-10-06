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
gap="$1"
case "$gap" in
  friction015) d_first=37101; d_second=37201; valid_first=37301; expert_first=37401; train_first=37501 ;;
  gain070) d_first=38101; d_second=38201; valid_first=38301; expert_first=38401; train_first=38501 ;;
  *) echo "Expected friction015 or gain070" >&2; exit 2 ;;
esac
repo="$portfolio_root"
root="$artifact_root/module_combo_v1/$gap"
python_bin="${SIMDIST_PYTHON:-python}"
m0="$checkpoint_root/models/go2_2026-02-12"
mkdir -p "$root/logs" "$root/dynamics/collections" "$root/dynamics/training" \
  "$root/expert/collections" "$root/policy" "$root/validation/m0" "$root/evaluations"

collect_d() {
  local checkpoint="$1" seed="$2" steps="$3" out="$4"
  if [[ ! -f "$out/summary.json" ]]; then
    "$python_bin" -B "$portfolio_root/experiments/module_combo_v1/collect_dynamics.py" \
      --gap "$gap" --checkpoint "$checkpoint" --seed "$seed" --steps "$steps" \
      --out "$out" > "$root/logs/collect_d_$(basename "$(dirname "$out")")_$seed.log" 2>&1
  fi
}

for stage in 10k 30k; do
  if [[ "$stage" == 10k ]]; then
    first="$d_first"; previous="$m0"
  else
    first="$d_second"; previous="$root/dynamics/training/10k/checkpoint"
  fi
  for offset in {0..5}; do
    seed=$((first + offset))
    if [[ "$stage" == 10k ]]; then
      steps=1667; if [[ "$offset" -ge 4 ]]; then steps=1666; fi
    else
      steps=3333; if [[ "$offset" -lt 2 ]]; then steps=3334; fi
    fi
    collect_d "$previous" "$seed" "$steps" "$root/dynamics/collections/$stage/seed_$seed"
  done
  echo "COLLECTED $gap $stage"

  train_paths=()
  heldout_paths=()
  for offset in {0..4}; do
    seed=$((d_first + offset))
    train_paths+=("$root/dynamics/collections/10k/seed_$seed/rollout.h5")
  done
  heldout_paths+=("$root/dynamics/collections/10k/seed_$((d_first + 5))/rollout.h5")
  latest="${heldout_paths[0]}"
  if [[ "$stage" == 30k ]]; then
    for offset in {0..4}; do
      seed=$((d_second + offset))
      train_paths+=("$root/dynamics/collections/30k/seed_$seed/rollout.h5")
    done
    heldout_paths+=("$root/dynamics/collections/30k/seed_$((d_second + 5))/rollout.h5")
    latest="${heldout_paths[1]}"
  fi
  out="$root/dynamics/training/$stage"
  train_seed="$train_first"
  if [[ "$stage" == 30k ]]; then train_seed=$((train_first + 1)); fi
  if [[ ! -f "$out/summary.json" ]]; then
    "$python_bin" -B "$portfolio_root/experiments/extended_adaptation_v1/train_seed_split.py" \
      --previous "$previous" --seed "$train_seed" \
      --train-rollouts "${train_paths[@]}" --heldout-rollouts "${heldout_paths[@]}" \
      --latest-heldout "$latest" --out "$out" > "$root/logs/train_d_$stage.log" 2>&1
  fi
  echo "TRAINED $gap $stage"
done

expert_train=()
for offset in {0..5}; do
  seed=$((expert_first + offset))
  steps=1667; if [[ "$offset" -ge 4 ]]; then steps=1666; fi
  out="$root/expert/collections/seed_$seed"
  if [[ ! -f "$out/summary.json" ]]; then
    "$python_bin" -B "$portfolio_root/experiments/module_combo_v1/collect_expert.py" \
      --gap "$gap" --seed "$seed" --steps "$steps" --out "$out" \
      > "$root/logs/collect_expert_$seed.log" 2>&1
  fi
  if [[ "$offset" -lt 5 ]]; then expert_train+=("$out/rollout.h5"); else heldout="$out/rollout.h5"; fi
done
echo "COLLECTED $gap expert"

for stage in m0 30k; do
  if [[ "$stage" == m0 ]]; then previous="$m0"; else previous="$root/dynamics/training/30k/checkpoint"; fi
  out="$root/policy/$stage"
  if [[ ! -f "$out/summary.json" ]]; then
    "$python_bin" -B "$portfolio_root/experiments/module_combo_v1/train_policy_50.py" \
      --previous "$previous" --train-rollouts "${expert_train[@]}" \
      --heldout-rollout "$heldout" --out "$out" > "$root/logs/train_p_$stage.log" 2>&1
  fi
  echo "TRAINED $gap P $stage"
done

for offset in {0..9}; do
  seed=$((valid_first + offset))
  collect_d "$m0" "$seed" 500 "$root/validation/m0/seed_$seed"
done
if [[ ! -f "$root/losses/c_loss_summary.csv" ]]; then
  "$python_bin" -B "$portfolio_root/analysis/module_combo_v1/evaluate_losses.py" --gap "$gap" \
    > "$root/logs/evaluate_losses.log" 2>&1
fi
echo "VALIDATED $gap losses"

seeds=()
for ((seed=23501; seed<=23550; seed++)); do seeds+=("$seed"); done
for name in n0p0_mppi d30p0_mppi n0p50_mppi d30p50_mppi n0p0_base n0p50_base; do
  case "$name" in
    n0p0_mppi) controller=mppi; checkpoint="$m0" ;;
    d30p0_mppi) controller=mppi; checkpoint="$root/dynamics/training/30k/checkpoint" ;;
    n0p50_mppi) controller=mppi; checkpoint="$root/policy/m0/checkpoint" ;;
    d30p50_mppi) controller=mppi; checkpoint="$root/policy/30k/checkpoint" ;;
    n0p0_base) controller=base; checkpoint="$m0" ;;
    n0p50_base) controller=base; checkpoint="$root/policy/m0/checkpoint" ;;
  esac
  "$python_bin" -B "$portfolio_root/experiments/followup_v1/eval_gap_batch.py" \
    --gap "$gap" --controller "$controller" --forward-vel 1.0 \
    --seeds "${seeds[@]}" --checkpoint "$checkpoint" \
    --out-root "$root/evaluations/$name" > "$root/logs/eval_$name.log" 2>&1
  echo "EVALUATED $gap $name"
done
