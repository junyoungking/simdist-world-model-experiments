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
root="$artifact_root/module_combo_v1/followup_repro"
py="${SIMDIST_PYTHON:-python}"
m0="$checkpoint_root/models/go2_2026-02-12"

run_d_chain() {
  local gap="$1" d1="$2" d2="$3" val="$4" train_seed="$5"
  local out="$root/$gap/second_chain"
  mkdir -p "$out/collections/10k" "$out/collections/30k" "$out/training" "$out/validation/m0" "$out/evaluations"
  for stage in 10k 30k; do
    local first previous
    if [[ "$stage" == 10k ]]; then first="$d1"; previous="$m0"; else first="$d2"; previous="$out/training/10k/checkpoint"; fi
    for offset in {0..5}; do
      local seed steps dest
      seed=$((first + offset)); steps=1667
      if [[ "$stage" == 10k && "$offset" -ge 4 ]]; then steps=1666; fi
      if [[ "$stage" == 30k ]]; then
        steps=3333; if [[ "$offset" -lt 2 ]]; then steps=3334; fi
      fi
      dest="$out/collections/$stage/seed_$seed"
      if [[ ! -f "$dest/summary.json" ]]; then
        "$py" -B "$portfolio_root/experiments/module_combo_v1/collect_dynamics.py" --gap "$gap" --checkpoint "$previous" \
          --seed "$seed" --steps "$steps" --out "$dest" > "$out/collect_${stage}_${seed}.log" 2>&1
      fi
    done
    local train_paths=() heldout_paths=() latest train_id
    for offset in {0..4}; do train_paths+=("$out/collections/10k/seed_$((d1 + offset))/rollout.h5"); done
    heldout_paths+=("$out/collections/10k/seed_$((d1 + 5))/rollout.h5")
    latest="${heldout_paths[0]}"
    if [[ "$stage" == 30k ]]; then
      for offset in {0..4}; do train_paths+=("$out/collections/30k/seed_$((d2 + offset))/rollout.h5"); done
      heldout_paths+=("$out/collections/30k/seed_$((d2 + 5))/rollout.h5")
      latest="${heldout_paths[1]}"
      train_id=$((train_seed + 1))
    else
      train_id="$train_seed"
    fi
    if [[ ! -f "$out/training/$stage/summary.json" ]]; then
      "$py" -B "$portfolio_root/experiments/extended_adaptation_v1/train_seed_split.py" --previous "$previous" --seed "$train_id" \
        --train-rollouts "${train_paths[@]}" --heldout-rollouts "${heldout_paths[@]}" \
        --latest-heldout "$latest" --out "$out/training/$stage" \
        > "$out/train_${stage}.log" 2>&1
    fi
    echo "CHAIN_TRAINED $gap $stage"
  done

  for offset in {0..9}; do
    local seed dest
    seed=$((val + offset)); dest="$out/validation/m0/seed_$seed"
    if [[ ! -f "$dest/summary.json" ]]; then
      "$py" -B "$portfolio_root/experiments/module_combo_v1/collect_dynamics.py" --gap "$gap" --checkpoint "$m0" \
        --seed "$seed" --steps 500 --out "$dest" > "$out/collect_validation_${seed}.log" 2>&1
    fi
  done
  "$py" -B "$portfolio_root/experiments/module_combo_v1/evaluate_latent_datasets.py" \
    --model "m0=$m0" --model "chain1_30k=$artifact_root/module_combo_v1/$gap/dynamics/training/30k/checkpoint" \
    --model "chain2_30k=$out/training/30k/checkpoint" \
    --dataset "independent_m0=$out/validation/m0" --out "$out/losses.csv" \
    > "$out/losses.log" 2>&1

  local ext=() all=()
  for ((seed=23551; seed<=23600; seed++)); do ext+=("$seed"); done
  for ((seed=23501; seed<=23600; seed++)); do all+=("$seed"); done
  "$py" -B "$portfolio_root/experiments/followup_v1/eval_gap_batch.py" --gap "$gap" --controller mppi --forward-vel 1.0 \
    --seeds "${ext[@]}" --checkpoint "$m0" --out-root "$out/evaluations/m0_extension" \
    > "$out/eval_m0_extension.log" 2>&1
  "$py" -B "$portfolio_root/experiments/followup_v1/eval_gap_batch.py" --gap "$gap" --controller mppi --forward-vel 1.0 \
    --seeds "${ext[@]}" --checkpoint "$artifact_root/module_combo_v1/$gap/dynamics/training/30k/checkpoint" \
    --out-root "$out/evaluations/chain1_30k_extension" > "$out/eval_chain1_extension.log" 2>&1
  "$py" -B "$portfolio_root/experiments/followup_v1/eval_gap_batch.py" --gap "$gap" --controller mppi --forward-vel 1.0 \
    --seeds "${all[@]}" --checkpoint "$out/training/30k/checkpoint" \
    --out-root "$out/evaluations/chain2_30k" > "$out/eval_chain2.log" 2>&1
  echo "CHAIN_EVALUATED $gap"
}

run_d_chain friction015 39101 39201 39301 39501
run_d_chain gain070 40101 40201 40301 40501

mass="$root/mass8_d_p"
mkdir -p "$mass/collections/10k" "$mass/collections/30k" "$mass/training" "$mass/validation/m0p" "$mass/evaluations"
policy_m0="$artifact_root/module_combo_v1/mass8/policy/m0/checkpoint"
for stage in 10k 30k; do
  if [[ "$stage" == 10k ]]; then first=42101; previous="$policy_m0"; train_id=42501
  else first=42201; previous="$mass/training/10k/checkpoint"; train_id=42502; fi
  for offset in {0..5}; do
    seed=$((first + offset)); steps=1667
    if [[ "$stage" == 10k && "$offset" -ge 4 ]]; then steps=1666; fi
    if [[ "$stage" == 30k ]]; then steps=3333; if [[ "$offset" -lt 2 ]]; then steps=3334; fi; fi
    dest="$mass/collections/$stage/seed_$seed"
    if [[ ! -f "$dest/summary.json" ]]; then
      "$py" -B "$portfolio_root/experiments/module_combo_v1/collect_dynamics.py" --gap mass8 --checkpoint "$previous" \
        --seed "$seed" --steps "$steps" --out "$dest" > "$mass/collect_${stage}_${seed}.log" 2>&1
    fi
  done
  train_paths=() heldout_paths=() latest=""
  for offset in {0..4}; do train_paths+=("$mass/collections/10k/seed_$((42101 + offset))/rollout.h5"); done
  heldout_paths+=("$mass/collections/10k/seed_42106/rollout.h5"); latest="${heldout_paths[0]}"
  if [[ "$stage" == 30k ]]; then
    for offset in {0..4}; do train_paths+=("$mass/collections/30k/seed_$((42201 + offset))/rollout.h5"); done
    heldout_paths+=("$mass/collections/30k/seed_42206/rollout.h5"); latest="${heldout_paths[1]}"
  fi
  if [[ ! -f "$mass/training/$stage/summary.json" ]]; then
    "$py" -B "$portfolio_root/experiments/extended_adaptation_v1/train_seed_split.py" --previous "$previous" --seed "$train_id" \
      --train-rollouts "${train_paths[@]}" --heldout-rollouts "${heldout_paths[@]}" \
      --latest-heldout "$latest" --out "$mass/training/$stage" > "$mass/train_${stage}.log" 2>&1
  fi
  echo "DP_TRAINED mass8 $stage"
done

for offset in {0..9}; do
  seed=$((42301 + offset)); dest="$mass/validation/m0p/seed_$seed"
  if [[ ! -f "$dest/summary.json" ]]; then
    "$py" -B "$portfolio_root/experiments/module_combo_v1/collect_dynamics.py" --gap mass8 --checkpoint "$policy_m0" \
      --seed "$seed" --steps 500 --out "$dest" > "$mass/collect_validation_${seed}.log" 2>&1
  fi
done

dp30="$mass/training/30k/checkpoint"
original_policy="$checkpoint_root/models/go2_2026-02-12"
"$py" -B "$portfolio_root/experiments/module_combo_v1/compose_policy_checkpoint.py" --dynamics "$dp30" \
  --policy "$original_policy" --out "$mass/composed_d_pnone" > "$mass/compose.log" 2>&1

"$py" -B "$portfolio_root/experiments/module_combo_v1/evaluate_latent_datasets.py" \
  --model "D=$artifact_root/extended_adaptation_v1/training/30k_c/checkpoint" \
  --model "D_P=$dp30" \
  --dataset "original_policy_mppi=$artifact_root/score_probe_v1/validation_collections/m0" \
  --dataset "m0_P_mppi=$mass/validation/m0p" \
  --dataset "expert_heldout=$artifact_root/extended_adaptation_v1/oracle/collections/seed_33806/rollout.h5" \
  --out "$mass/losses.csv" > "$mass/losses.log" 2>&1

seeds=(); for ((seed=23501; seed<=23600; seed++)); do seeds+=("$seed"); done
"$py" -B "$portfolio_root/experiments/followup_v1/eval_gap_batch.py" --gap mass8 --controller mppi --forward-vel 1.0 \
  --seeds "${seeds[@]}" --checkpoint "$dp30" --out-root "$mass/evaluations/d_p_mppi" \
  > "$mass/eval_d_p.log" 2>&1
"$py" -B "$portfolio_root/experiments/followup_v1/eval_gap_batch.py" --gap mass8 --controller mppi --forward-vel 1.0 \
  --seeds "${seeds[@]}" --checkpoint "$mass/composed_d_pnone/checkpoint" \
  --out-root "$mass/evaluations/d_pnone_mppi" > "$mass/eval_d_pnone.log" 2>&1
echo "FOLLOWUP_REPRO_ALL_COMPLETE"
