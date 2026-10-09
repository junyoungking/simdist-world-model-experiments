# October 9 portfolio raw evidence provenance

The builder reads saved seed/episode/state measurements, with reports and
README used only as comparison expectations. It does not import experiment
code, run a model/simulator, or change original research files.

```bash
python -B analysis/build_portfolio_evidence_from_raw.py \
  --raw-root "$SIMDIST_RAW_ROOT" \
  --out results/portfolio_evidence_20261009 --check
python -B analysis/build_portfolio_evidence_from_raw.py \
  --raw-root "$SIMDIST_RAW_ROOT" \
  --out results/portfolio_evidence_20261009 --write
```

Run from this repository with NumPy and SciPy installed. `SIMDIST_RAW_ROOT`
is the original `scratch` directory. Original checkpoint directories must
also be available in their historical layout relative to that directory;
base checkpoints are in the adjacent `checkpoints` directory. Historical
machine prefixes are normalized at runtime; public metadata uses relative
sources and logical checkpoint IDs.

`--check` writes nothing and compares all six compact CSVs, generated
provenance, actual README table values and historical numeric expectations.
`--write` recomputes and validates data before publishing. Each existing CSV
has an independent compatibility gate; an unexpected mismatch preserves that
file and returns a failure. Other validated files can be published. The known
mass validation-family mix-up and the explicit removal of the unsupported p
are authorized, recorded migrations, not suppressed validation failures.
Missing/duplicate seed, malformed schema, nonfinite value, wrong condition,
loss dataset mismatch or changed input blocks publication.

The six original CSV headers and row counts are retained. Displayed loss/rho/D
values keep the original precision; exact raw-derived values, model labels,
denominators and SHA256 are in the generated
[PROVENANCE.json](../results/portfolio_evidence_20261009/PROVENANCE.json).
`source_report_section` remains as a compatibility field and now directs
readers to raw-derived provenance. It is never an aggregation input.

| File | Compact rows | Detailed measurement rows | Status |
|---|---:|---:|---|
| mass_chains_200seed.csv | 7 | 7 | Reproduced from raw episodes and independent seed MSE |
| chain_equality_test.csv | 1 | 0 | NOT_REPRODUCED status only; p empty |
| value_mc_rank.csv | 4 | 4 | Historical175/200 horizons preserved |
| common_state_dynamics_error.csv | 4 | 4 | Shared selected states; model-specific completion masks |
| target_failure_summary.csv | 4 | 18 | Target held-out seed means; mass interval corrected |
| flat_velocity_comparison.csv | 2 | 5 | Training speed1.0 and0.75 groups evaluated at0.75 |

## Statistical reproduction boundary

Code, reports and result metadata in the requested source/portfolio trees
were searched for the original six-chain test. An additional86 top-level
producer/report/result/log files in positive_control, chain_variance,
multi_gap and gap_diagnosis were inspected and hashed locally. The only
approximate-p keyword match there was an unrelated torque reward weight.
The original statistic, test implementation and exact p were not found.
The old compact CSV expressly described its p as report transcription.

The six adapted chains have the same200 evaluation seeds. They comprise four
P0-collected chains and two policy-modified-data chains, with different
adaptation histories/configuration; those200 paired evaluation seeds are not
six independent training repetitions, and the1200 outcomes are not treated
as independent observations. No replacement test was performed.
`chain_equality_test.csv` is generated as a status record: empty
`approx_p_value`, `evidence_status=NOT_REPRODUCED`. Its detailed provenance
contains only the verified paired coverage and per-chain failure counts.
README removes the historical rejection claim and unverified100-seed pairwise
p claim. Recovering the original method remains necessary to reproduce either
historical statistical claim. This absence does not block independently
verified measurement tables and is never presented as a reproduced p-value.

## Loss families and MC protocols

Section4 mass uses ten independent M0 collection seeds49301–49310 and an
arithmetic mean of per-seed prediction MSE. The four P0 adaptation chains
reduce that loss by about31–45%, relative to the identical M0 validation set.
Section7 uses each target's common held-out collection seeds186001–186020,
again an arithmetic seed mean. Mass reductions are32.299027–46.356004%,
rounding to32–46%. The previous target CSV's31–45 interval mixed these two
families and is corrected by calculation, not manual CSV editing.
`loss_n` is the seed-mean denominator, `windows` is the total number of valid
windows, and `loss25_window_pooled` is separately identified as a weighted
window average. Headline loss/reduction always uses `loss25` seed means.
Evaluation n200 is separate from loss n10/n20 and D/MC snapshot counts.

The historical source/mass expert continuation is175 steps after step25;
friction/gain continuation is200. Head and critic use identical within-target
MC pairs, discount0.99, pooled snapshots and the same exclusion mask. Original
headline correlations are preserved. Cross-target MC returns are not directly
compared; horizons are disclosed in README, CSV evaluation_basis and provenance.
No trimming, survivor re-filtering or October8–9 replacement was introduced.

## Discovered schemas and exact selections

### Episode evaluation and tile first-exit

Inputs are physical `summary.json`, sibling `evaluation_steps.csv` and
`warmup_steps.csv`. Summary fields include `seed`, `status`, `controller`,
`mass_added_kg`, `forward_target_mps`, `physical_readback`, reset errors,
`warmup_steps`, `episode_length`, `fall` and `return`. Later summaries also
record `gap`, `terrain` and `foot_friction`; older ones omit terrain/friction.
The legacy omissions use the **explicitly discovered** historical
`go2_probe_common.py` configuration, checked against
`positive_control_variance_v1/source_hashes_before.json`. Friction/gain physical
readback supplies their changed material/gains. Source pin mismatch is a failure.

Step CSV fields used are `step`, `reward`, `terminated`, `command_vx`. Each
model must contain exactly the two disjoint seed sets **23501–23600** and
**24001–24100**. Duplicate seeds, missing block coverage, nonfinite values,
nonbinary flags, nonsequential steps, unfinished episodes, nonterminal short
episodes, inconsistent return, reset errors or mismatched conditions fail.
Episode length is at most1000, excluding expert warmup25. Full-episode failure
is physical termination or `warmup_failed`.

Mass uses seven models: m0, old_chain1, old_chain2, new_chain3, new_chain4,
old_DP, new_DP2. The old100 m0 block is split across
`results/s1_v2/runs` (23501–23510, only `mppi_mass8_v10_seed*`),
`adaptation_curve_v1/results/evaluations/baseline_target` (23511–23520, MPPI
only), and `score_probe_v1/results/evaluations/m0` (23521–23600).
old_chain1 is split between stage_30000 (23501–23520) and cp30000
(23521–23600); old_chain2 comes from followup evaluation; old_DP comes from
the module-composition `d_pnone_mppi` evaluations. The three newer chains' old
seeds come from positive-control physical evaluations. New100 m0/old_DP/new_DP2
come from `chain_variance_recheck_v1/results/evaluations/mass8_new_*`; the
remaining new100 come from `multi_gap_200seed_extension_v1/results/evaluations`.

Friction/gain old100 m0 and chain1 use module-composition evaluations for
23501–23550 and second_chain extension evaluations for23551–23600. Their
chain2 uses second_chain/chain2_30k for all old100. New100 and friction60k/90k
come from multi-gap physical evaluations. Friction60k/90k each have all200
physical episodes; the retained raw directory contents, not an old scheduling
decision, determine coverage. Exact directories/ranges are encoded in
`episode_sources`, not inferred from a report at runtime.

For boxes targets, reproduce `tile_failure_data_reanalysis_v1/analyze.py`:
start clock at float32 zero, accumulate float32 `.02` per tick, and reconstruct
pre-action displacement at CSV step k as
`clock[25+k-1] - command_vx`. The first observed displacement **greater than
7.425m** is exit. A later termination remains tile success; a termination before
observed exit is tile failure. Warmup failure or an unfailed right-censored
episode with no exit blocks this calculation. This is an observed control-tick
forward-displacement rule, not continuous trajectory crossing or lateral exit.

### Held-out prediction losses

Section4 input:
`positive_control_variance_v1/results/mass8/losses.csv`. Its columns are
`model, checkpoint, checkpoint_step, dataset, rollout, windows,
one_step_latent_mse, full_25_step_latent_mse`.
Select `dataset=independent_m0`, all seven mass models, exactly ten distinct
rollout seeds49301–49310. These are persisted per-rollout prediction
measurements from `module_combo_v1/evaluate_latent_datasets.py`, not the
aggregated `losses_summary.csv`. Underlying rollout files must exist and are
hashed. Model validation references and window counts must agree exactly.

Section7 boxes input:
`common_state_gap_v1/results/loss/{target}/{model}/seed_*/summary.json`,
fields `target,model,seed,windows,loss1,loss25`. Select exactly186001–186020;
target/model/seed identities and per-seed window counts must agree across
models. The shared collection's `rollout.h5` files are hashed. Use each target's
own M0 collection; do not pool across targets or substitute section4 losses.

Matched flat input: `flat_velocity_matched_v1/results/losses.csv`, same
per-rollout schema as section4. Select `dataset=common075`, seeds186001–186020,
models m0/flat075_chain1/flat075_chain2. The separate `independent100` set is
not substituted. The selected common collection command is0.75m/s.

Every reported loss is the **arithmetic mean of independent seed-level MSEs**;
sample SD and seed n are retained. Section4 additionally exposes window-weighted
pooled loss as a separately named field, never as the headline mean. Reduction
is `100*(1-model_seed_mean/M0_seed_mean)` on the identical target validation
set. The source HDF5 observations are never passed to a model by this builder.

### Value head/privileged critic versus MC

Inputs are raw `candidates.csv` and `sequences_with_plans.npz` (if present),
otherwise `sequences.npz`, under source/mass
`candidate_rank_v1/results/full/m0_mass0` and `m0_mass8`, and friction/gain
`gap_diagnosis_v1/results/branches/friction_m0` and `gain_m0`.

CSV fields used: `candidate_kind,seed,snapshot_step,actual_steps,
expert_critic_step25`. Array keys used: `candidate_kinds,actual_head_valid,
actual_env_reward_sequence,actual_latent_value_head_sequence`. Require exactly
one `expert_closed` branch, matching array/CSV candidate order and unique
seed/snapshot identity. Exclude incomplete25 expert heads by the saved validity
flag. Both correlations use the same retained snapshot pairs. Compare head
value at step25 and critic at step25 with the subsequent env reward sequence,
discount0.99 restarted at continuation step0. Pool snapshots, then apply
Spearman with SciPy's tied ranks; do not average per-seed correlations.

| Target | Valid/selected snapshots | Head rho | Critic rho | Continuation horizon |
|---|---:|---:|---:|---:|
| Source | 144/145 | 0.5660075557 | 0.7641065831 | 175 |
| Mass8 | 95/97 | 0.5651035834 | 0.7737122060 | 175 |
| Friction015 | 146/150 | 0.3901921279 | 0.6791300743 | 200 |
| Gain070 | 147/150 | 0.0538840490 | 0.4366825590 | 200 |

Exact selected counts are also verified in the JSON. No October8–9 MC headline
or separate 34-candidate source dynamics result is introduced.

### Common-state dynamics reward error

Select states from original
`common_state_gap_v1/results/loss/{target}/{model}/seed_*/states.json` for
186001–186020. Fields used: `state_id,seed,state_path,step,episode`. Every model
inside a target must select the same complete state identities and saved-state
references. Match each to
`results/open/{target}/{model}/seed_{seed}/{state_id}/summary.json` and
`mean_plan.npz`. Validate branch kind/identity/completion and exact executed
actions against the stored plan. Array fields are `actions,plan,
predicted_rewards,actual_rewards_head`.

Recompute `D=sum(0.99**t * predicted_rewards[t]) -
sum(0.99**t * actual_rewards_head[t])` for t0–24. Stored summary D is only a
cross-check (tolerance3e-6 for float32 producer versus float64 reaggregation).
Exclude terminated or incomplete25 branches as the original method does.
Take the state mean, then the min/max of adapted model means within the target.
All selected states are shared, but **valid completion masks are model-specific**;
the model coverage JSON records their different n. `mean_plan` is the same
selection rule, not an identical action vector across checkpoints.

| Target | Selected states | M0 valid states | Models included |
|---|---:|---:|---|
| mass8 | 192 | 183 | seven mass models |
| friction015 | 183 | 156 | m0, chain1, chain2,60k,90k |
| gain070 | 195 | 192 | m0, chain1,chain2 |
| flat | 188 | 153 | m0, flat_chain1,flat_chain2 |

Flat checkpoints trained with command1.0m/s use the shared evaluation collection
at0.75m/s. The source candidate diagnostic D=0.098 is excluded.

### Flat velocity comparison

Source m0 and original1.0-trained flat_chain1/2 physical evaluations are under
`multi_gap_200seed_extension_v1/results/evaluations/flat075_*`, all200 seeds.
Retrained0.75 flat075_chain1/2 are under
`flat_velocity_matched_v1/results/evaluations/v075/flat075_chain*`, all200.
Use full episode failure, evaluate0.75 for every row. Named raw results are
M0=53%, flat_chain1=91.5%, flat_chain2=94.5%, flat075_chain1=62%,
flat075_chain2=58%. The compact CSV/README's unnamed `58;62` list does not
assign that sorted order to chain1/2.

## Public scope and local audit evidence

[The public audit](portfolio_raw_audit_20261009.json) records current execution
and validation summaries. Generated PROVENANCE.json includes38 detailed
measurement rows, one missing-test status, explicit producer/per-seed loss
sources with full SHA256, grouped input-set digests and logical checkpoint
IDs/steps/full checkpoint-tree digests. It does not enumerate per-episode or
per-state input filenames. These are the minimum provenance needed to explain
selection and bind the measurements without publishing the entire inventory.

The original20663-file audit, original six compact CSVs, source-search details
and the new full input/checkpoint manifest remain in ignored
`analysis/.local_audit/`. They are local evidence, not commit candidates.
The current raw input count is20665: the original inputs plus two explicit
checkpoint/condition manifests. Checkpoint trees are additionally hashed
without loading model parameters or copying checkpoint data. A checkpoint tree
hash is SHA256 over sorted checkpoint-relative filename TAB file-SHA256 LF;
raw input-set hashes use raw-root-relative filename TAB file-SHA256 LF.
Public metadata exposes these digests, not the individual filenames.

Actual source schemas and checkpoint/dataset bindings are validated. Source
code is hashed but never executed. Full hashes and size/mtime/inode checks,
repeated real-data regeneration, and unchanged tracked source Git status
provide current-run integrity evidence. They do not claim certification of
unrecorded historical training seeds or exact PhysX replay.

README comparison parses current numeric tables and exact labels; no whole
README hash bypass is needed for authorized wording/whitespace changes.
Static verification is separate from raw measurement validation and the
unreproduced statistic. The execution summary reports
`VALIDATED_WITH_NOT_REPRODUCED_TEST`, not a reproduced statistical test.
No experiments, raw/checkpoint edits, commit or push are performed.

## Executed controls and review list

Real-data check is read-only and returns0 after generation. Repeating write
with identical inputs produces byte-identical six CSVs and PROVENANCE.json,
and identical full source/checkpoint digests. Synthetic controls verify that
an unexpected CSV mismatch preserves that file and returns failure while
other tables validate, duplicate loss seeds/nonfinite values fail, seed mean
and window-pooled mean remain distinct, and absent test evidence cannot be
published as a numeric p. No synthetic values enter the evidence tables.

README's41 relative links, all six legacy headers/row counts,66 historical
numeric expectations,39 actual README numeric cells, private-path/credential
scan and git whitespace check pass. Original tracked source status and
registered raw/checkpoint hashes remain unchanged. `tools/verify_portfolio.py`
is STATIC_PASS; statistical reproduction is still NOT_REPRODUCED.

The exact15 commit candidates are listed in the public audit JSON. Review:

```bash
git diff -- README.md docs/REPRODUCIBILITY.md docs/VALIDATION.json .gitignore
git diff --no-index /dev/null analysis/build_portfolio_evidence_from_raw.py
git add -- .gitignore README.md docs/REPRODUCIBILITY.md docs/VALIDATION.json \
  analysis/build_portfolio_evidence_from_raw.py \
  analysis/PORTFOLIO_RAW_PROVENANCE.md analysis/portfolio_raw_audit_20261009.json \
  results/portfolio_evidence_20261009/mass_chains_200seed.csv \
  results/portfolio_evidence_20261009/chain_equality_test.csv \
  results/portfolio_evidence_20261009/value_mc_rank.csv \
  results/portfolio_evidence_20261009/common_state_dynamics_error.csv \
  results/portfolio_evidence_20261009/target_failure_summary.csv \
  results/portfolio_evidence_20261009/flat_velocity_comparison.csv \
  results/portfolio_evidence_20261009/PROVENANCE.json \
  results/portfolio_evidence_20261009/README.md
git commit -m "Regenerate portfolio measurements from raw evidence; mark historical test unreproduced"
```

These commands were prepared for the user and were not executed. Never add
`analysis/.local_audit/`; its full inventory and preserved original CSVs are
local evidence. No commit/push or new experiment was performed.
