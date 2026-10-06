# SimDist World-Model Experiments

Research experiments on action-conditioned prediction, dynamics adaptation, and downstream planning/control under a Go2 dynamics shift.

## My Contributions

- Built data collection, world-model adaptation, held-out prediction, and downstream MPPI evaluation pipelines.
- Designed large-scale matched-seed evaluations and diagnostics connecting action-conditioned prediction to control behavior.
- Investigated candidate ranking errors, adaptation-chain variance, model/policy composition, and negative or inconclusive hypotheses.

The world-model architecture, core MPPI controller, released checkpoints, and base simulator integration come from [CLeARoboticsLab/SimDist](https://github.com/CLeARoboticsLab/simdist). My contribution is the additional experiment pipelines, controlled comparisons, diagnostics, and analysis in this repository. The two upstream patches add instrumentation; they do not constitute a new implementation of the upstream core algorithm.

This is a public **code/evidence subset**, with 151 selected source artifacts and two instrumentation patches. It is not the complete historical binary bundle. Checkpoints, rollout HDF5, large candidate arrays, smoke outputs, superseded results, failed T3A n10, and the ongoing gap-diagnosis run are excluded. See [reproducibility and dependencies](docs/REPRODUCIBILITY.md), the [source/public hash manifest](docs/BUILD_MANIFEST.json), and the [static validation report](docs/VALIDATION.json).

## 1. Action-Conditioned World Model Adaptation

The main setting adds 8 kg to the simulated Go2 base. Adaptation updates the dynamics component **f only**, while the other learned components remain fixed. Collection budgets span 1k, 3k, 10k, 30k, 60k, and 90k interactions. The experiments compare held-out latent prediction with downstream MPPI control.

| Checkpoint | Mean return | Failures / 100 evaluation seeds |
|---|---:|---:|
| M0 | 16.686 | 62 |
| 10k | 22.951 | 50 |
| 30k | 25.123 | 40 |
| 90k | 25.748 | 37 |

The 100-seed evaluation extends 20 existing episodes with 80 additional seeds. It is not 100 independently trained models. Control gains after 30k were not monotonic: the 30k_c/60k/90k sequence and independent adaptation chains show why lower prediction error alone is insufficient to select a controller.

- Code: [collection](experiments/adaptation_curve_v1/collect.py), [adaptation](experiments/adaptation_curve_v1/train_stage.py), [seed-held-out adaptation](experiments/extended_adaptation_v1/train_seed_split.py), [control evaluation](experiments/score_probe_v1/eval_probe_batch.py), [independent latent-MSE](analysis/score_probe_v1/independent_loss.py).
- Evidence: [100-seed summary](results/score_probe_v1/analysis/a_conditions.csv), [seed-level control](results/score_probe_v1/analysis/a_seed_results.csv), [independent loss summary](results/score_probe_v1/analysis/c_loss_summary.csv), [extended control](results/extended_adaptation_v1/aggregate/condition_summary.csv), [clean independent loss](results/extended_adaptation_v1/aggregate/independent_loss_clean.csv).

## 2. Prediction ≠ Control / Candidate Ranking

The planning diagnostic executes candidate actions from saved states and compares predicted ranking with observed rollout outcomes. It covers **371 snapshots, 25,228 candidate branches, and 24,424 full-25-step branches**. Snapshot-level summaries and reconstruction audits are included; the large branch arrays remain external.

PhysX's complete internal state was not restored. Assumed and realized command paths also differ. The reward-head/environment and prediction/observation terms are diagnostic quantities, not a pure causal decomposition into **f / R / V**. The policy-specific critic and incomplete branches are treated explicitly.

- Code: [candidate execution](experiments/candidate_rank_v1/run_candidates.py), [state capture/restore](experiments/candidate_rank_v1/state.py), [model heads](experiments/candidate_rank_v1/model_heads.py), [ranking analysis](analysis/candidate_rank_v1/analyze.py), [decomposition](analysis/candidate_rank_v1/reanalyze_decomposition.py), [raw audit](analysis/candidate_rank_v1/audit_raw.py).
- Evidence: [371 snapshot metrics](results/candidate_rank_v1/analysis/per_snapshot_metrics.csv), [ranking summary](results/candidate_rank_v1/analysis/ranking_summary.csv), [expert rank summary](results/candidate_rank_v1/decomposition_v1/expert_rank_summary.csv), [raw coverage audit](results/candidate_rank_v1/analysis/raw_audit.json), [decomposition audit](results/candidate_rank_v1/decomposition_v1/audit.json), [definitions and limits](docs/candidate_decomposition.md).

## 3. Independent Adaptation-Chain Variance

Separately collected and trained mass8 adaptation chains produce different downstream outcomes: adapted-chain failures range from **37 to 62 / 100** on matched evaluation seeds. Fresh independent validation data supports the comparison of prediction errors across chains.

These are not fully identical iid training replicates. Collection policies, data splits, stage budgets, and initialization differ across parts of the lineage; the training and collection summaries preserve those differences. Shared evaluation seeds support paired comparisons rather than increasing the number of independent trained models.

- Code: [independent-chain driver](experiments/positive_control_variance_v1/run.py), [second-chain launcher](experiments/followup_v1/train_chain.sh), [cross-model latent loss](experiments/module_combo_v1/evaluate_latent_datasets.py), [aggregation](analysis/positive_control_variance_v1/aggregate.py).
- Evidence: [chain comparison](results/positive_control_variance_v1/mass8/aggregate/chain_comparison.csv), [control summary](results/positive_control_variance_v1/mass8/aggregate/condition_summary.csv), [700 seed-level rows](results/positive_control_variance_v1/mass8/aggregate/seed_results.csv), [independent loss](results/positive_control_variance_v1/mass8/losses_summary.csv), [training summary](results/positive_control_variance_v1/training_summary.csv), [historical report](docs/chain_variance/REPORT.md).

## 4. Dynamics / Policy Module Composition

The module experiments separately adapt dynamics and fit the policy head, then evaluate combinations under the same mass8 control contract.

| Combination | Mean return |
|---|---:|
| D30–P0 | 25.123 |
| M0–P50 | 24.049 |
| D30–P50 | 19.148 |

Each module's improvement did not guarantee better combined control in this comparison. This is a result for these learned modules, data, and evaluation contracts—not a universal law of model/policy composition. Follow-up D_P experiments further distinguish collection-policy and transplant effects.

- Code: [policy fitting](experiments/module_combo_v1/train_policy_50.py), [checkpoint composition](experiments/module_combo_v1/compose_policy_checkpoint.py), [mass8 launcher](experiments/module_combo_v1/run_mass8.sh), [analysis](analysis/module_combo_v1/aggregate.py).
- Evidence: [module summary](results/module_combo_v1/mass8/aggregate/condition_summary.csv), [800 seed-level rows](results/module_combo_v1/mass8/aggregate/seed_results.csv), [paired contrasts](results/module_combo_v1/mass8/aggregate/paired_summary.csv), [D_P follow-up](results/module_combo_v1/followup_repro/mass8_d_p/aggregate/condition_summary.csv), [historical report](docs/module_composition/REPORT.md).

## 5. Negative / Inconclusive Experiments

| Experiment | Latest interpretation | Evidence |
|---|---|---|
| T1 equal-total composition | n=5, confidence interval crosses zero: inconclusive | [raw contrasts](results/t1/DATA_AMOUNT_CONTROL_RAW.csv), [summary](results/t1/DATA_AMOUNT_CONTROL_SUMMARY.csv), [report](docs/t1/T1_DATA_AMOUNT_CONTROL_REPORT.md) |
| T2 common-control/reset validation | Protocol failure; not a scientific negative result from an eligible causal comparison | [sync gate](results/t2/T2V2_SYNC_GATE.csv), [action hashes/reset errors](results/t2/T2V2_PREFIX_ACTION_HASHES.csv), [report](docs/t2/T2V2_PAIRWISE_CAUSAL_REPORT.md) |
| T3 matched/distant support | n=5, command-specific confidence intervals cross zero: inconclusive | [n5 summary](results/t3/FRESH_REPLICATION_SUMMARY.csv), [contrasts](results/t3/FRESH_REPLICATION_CONTRASTS.csv), [protocol audit](results/t3/FRESH_REPLICATION_AUDIT.csv) |

The T3 support experiment is distinct from command-allocation replication. The failed allocation n10 is excluded. T1's incomplete downstream-control run is not presented here as a completed control study.

## 6. Privileged PPO Capability Gap

The historical mass8 comparison includes three trained PPO seeds and ten primary evaluation seeds per policy. The adapted-WM MPPI group mean is approximately **35.581**, privileged PPO approximately **49.293**, a difference of approximately **13.712**.

PPO uses privileged observations. The controller and reset contracts also differ, so this is a capability reference under the recorded contracts, not a pure algorithm comparison. These September results use a different contract from the October adaptation table above; the return values should not be pooled.

![Mass8 capability reference with source anchors and early-termination markers](figures/PERFORMANCE_GAP_PLOT.png)

- Code: [target PPO training](experiments/privileged_ppo/train_target.py), [PPO evaluation](experiments/privileged_ppo/evaluate_ppo.py), [MPPI evaluation](experiments/privileged_ppo/evaluate_mppi.py), [summary generation](analysis/privileged_ppo/summarize.py).
- Evidence: [model summary](results/privileged_ppo/MODEL_SEED_SUMMARY.csv), [70 primary cells](results/privileged_ppo/MASS8_PRIMARY_EVAL_RAW.csv), [training curves](results/privileged_ppo/PPO_TRAINING_CURVES.csv), [paired runtime audit](results/privileged_ppo/PAIRED_RUNTIME_CONTRACT_AUDIT.csv), [final report](docs/privileged_ppo/MASS8_GAP_COMPARISON.md), [historical final verification](docs/privileged_ppo/FINAL_VERIFICATION.json).

## Upstream Boundary and Reproduction

Use a separate external SimDist checkout at `b63f76ba01457b4df0954f88985a3a73a206ec17` and IsaacLab at `157c6b74ed4d9892c5e5ccc0d38d7835f27e98f9` (version 2.0.1). External checkpoints, rollout HDF5, and candidate arrays are required for experiment reruns. No upstream implementation or simulator tree is vendored here.

- [MPPI patch](patches/mppi.patch): score diagnostic instrumentation.
- [Trainer patch](patches/trainer.patch): logging / held-out evaluation / optimizer-count instrumentation, plus worker-zero loader compatibility.
- [Reproducibility contract](docs/REPRODUCIBILITY.md): environment variables, artifact layout, historical versions, and validation scope.
- [Upstream MIT notice](docs/licenses/SimDist_LICENSE.txt).

Hashes in historical provenance JSON describe the **historical artifact hash**, not the sanitized public copy. The public [build manifest](docs/BUILD_MANIFEST.json) records original source, initial-copy, and final public hashes separately. PPO's recorded execution version is `34b1738a857745d22ed37bea17de8d4065a014c8`; three source files subsequently changed for recovery before the audited source HEAD `c21f2ae314351832fba924674a7406a910de700a`. Public code adds path/import routing and is not byte-identical to either historical version.

Static validation does not run training, simulation, or evaluation:

```bash
python -B tools/verify_portfolio.py
```
