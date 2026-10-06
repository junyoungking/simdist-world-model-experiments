# Reproducibility and Dependency Contract

## Scope

This repository is a public code/evidence subset: selected experiment code, compact result tables, one representative figure, protocols, failure reports, and provenance. It is not the complete historical experiment binary bundle. The public CSV/JSON files allow inspection of the reported results. Most original aggregation, raw-audit, training, and simulator entry points also require external artifacts.

Repository creation used copying, path/import sanitization, and static validation. It did not run training, simulation, or evaluation. Original sources were not modified.

## Pinned External Implementations

| Dependency | Revision |
|---|---|
| [CLeARoboticsLab/simdist](https://github.com/CLeARoboticsLab/simdist) | `b63f76ba01457b4df0954f88985a3a73a206ec17` |
| IsaacLab | `157c6b74ed4d9892c5e5ccc0d38d7835f27e98f9` |
| IsaacLab VERSION | `2.0.1` |

SimDist supplies the world-model architecture, core MPPI algorithm, losses, data processors, model-loading code, and base Go2 environment. IsaacLab/IsaacSim supply the simulator and tasks. These implementations are not copied into this repository. The upstream [MIT notice](licenses/SimDist_LICENSE.txt) is retained.

The recorded runtime includes Python 3.10.21, IsaacSim 4.5.0.0, Torch 2.5.1+cu121, and RSL-RL 2.2.1. The SimDist package pins Flax 0.10.4, Hydra 1.3.2, h5py 3.15.1, JAX CUDA 0.6.0, NumPy 1.26.4, Optax 0.2.4, and Orbax checkpoint 0.11.25. SciPy, matplotlib, OmegaConf, YAML, tqdm, wandb, and simulator-provided `omni`/Isaac packages are also referenced. Static validation does not certify installation, CUDA availability, or GPU runtime compatibility.

Apply the two patches to a separate writable checkout of the pinned upstream base. Do not apply them to an already patched research checkout.

```bash
git -C "$SIMDIST_ROOT" apply --check "$PORTFOLIO_ROOT/patches/mppi.patch"
git -C "$SIMDIST_ROOT" apply --check "$PORTFOLIO_ROOT/patches/trainer.patch"
git -C "$SIMDIST_ROOT" apply "$PORTFOLIO_ROOT/patches/mppi.patch"
git -C "$SIMDIST_ROOT" apply "$PORTFOLIO_ROOT/patches/trainer.patch"
```

The MPPI patch adds score diagnostic instrumentation. The trainer patch adds logging, fixed held-out evaluation, optimizer invocation counts, and worker-zero prefetch compatibility. The core algorithm is upstream work.

## Roots and Imports

`portfolio_paths.py` routes paths and connects local helpers to real external packages. It does not stub or replace upstream APIs.

| Environment variable | Contract |
|---|---|
| `SIMDIST_ROOT` | Separate pinned checkout containing `simdist/` and `config/`; required by upstream-dependent entry points |
| `SIMDIST_CHECKPOINT_ROOT` | External checkpoint root; defaults to `$SIMDIST_ROOT/checkpoints` |
| `SIMDIST_ARTIFACT_ROOT` | External full-run artifact/output root; default is this repository's `results/` for evidence inspection |
| `SIMDIST_PYTHON` | Interpreter for shell/serial drivers; otherwise the selected Python/current interpreter |
| `SIMDIST_C_OUTPUT_ROOT` | Explicit output namespace for the historical collection/adaptation harness |
| `SIMDIST_C_EXPERIMENT_MODE` | Required harness mode: `mass`, `friction`, or `kp`; preserve the original protocol's mode |

For reruns, choose an artifact/output root separate from the checked-in evidence. Existing CLI arguments for checkpoints, rollouts, datasets, seeds, and output directories remain available. Routing changes do not change seed values, hyperparameters, controller settings, or scientific result values.

Example environment setup, to be adapted to installed external locations:

```bash
export PORTFOLIO_ROOT="$PWD"
export SIMDIST_ROOT="${EXTERNAL_SIMDIST_CHECKOUT:?Set the external checkout}"
export SIMDIST_CHECKPOINT_ROOT="${EXTERNAL_CHECKPOINTS:?Set the checkpoint root}"
export SIMDIST_ARTIFACT_ROOT="${EXTERNAL_EXPERIMENT_ARTIFACTS:?Set the artifact root}"
export PYTHONPATH="$PORTFOLIO_ROOT:$SIMDIST_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

The portable shell launchers calculate the portfolio root from their own location. Python entry points find `portfolio_paths.py` above themselves and register the selected common/data/candidate helper directories. They still require the real SimDist and simulator packages.

## External Artifacts

The following inputs are deliberately not included:

- Released Go2 WM checkpoint: `models/go2_2026-02-12`, M0 step 367004.
- RL actor, policy, and critic: `rl/go2_2026-02-11/model_4999.pt`, `policies/policy_4999.pt`, and `critics/critic_4999.pt` below that RL run.
- Generated adapted checkpoints and composed dynamics/policy checkpoints.
- Collection rollout HDF5 and processed prediction datasets/scalers.
- Large candidate branch arrays, state snapshots, and saved NPZ sequences.
- Full raw evaluation JSON/step records, PPO training/resume checkpoints, original hash inventories, and detailed runtime readback artifacts needed by historical final-verification scripts.

Use these normalized family directories below `SIMDIST_ARTIFACT_ROOT`:

```text
baseline_v2/runs/
adaptation_curve_v1/{collections,training,evaluations,aggregate}/
score_probe_v1/{validation_collections,evaluations,analysis}/
candidate_rank_v1/{full,analysis,decomposition_v1}/
extended_adaptation_v1/{collections,training,evaluations,oracle,aggregate}/
followup_v1/{chain,oracle_snapshots}/
module_combo_v1/{mass8,friction015,gain070,followup_repro}/
positive_control_variance_v1/{mass8,flat,screen}/
t1/{datasets,adaptation,models,eval}/
t3/{datasets,adaptation,evaluation,subsets,fresh_replication}/
t2/
privileged_ppo/{00_protocol,02_training,03_validation,04_0kg_anchor,
                05_mass8_primary,06_critic,07_summary,09_provenance}/
legacy/<historical-family>/
```

This is a layout contract, not a claim that those binaries are present. Preserve each supplied run's internal structure, manifests, and filename identities. The small public T1/T3/PPO tables use curated destination names; full-run audits expect the original subdirectory structure under the external artifact root.

Historical T1 composition requires the prior nonstationarity datasets. T3 subset construction requires the topic-continuation datasets and the prior support CSV/vector NPZ. T2 reset validation requires the original REC/CUM model inputs. These older roots are addressed as `legacy/<historical-family>/` and are not reconstructed from the small public summaries.

Metadata references are portable identifiers:

| Prefix | Resolution |
|---|---|
| `artifact://` | `SIMDIST_ARTIFACT_ROOT` |
| `checkpoint://` | `SIMDIST_CHECKPOINT_ROOT` |
| `portfolio://` | This repository |
| `simdist://` | `SIMDIST_ROOT` |

`resolve_artifact_path` resolves these identifiers when analysis reads a CSV/manifest path. `result_path` prefers a supplied full-run result and falls back to a checked-in public summary. `document_path` prefers the external original protocol and otherwise locates the public document. Historical hash verification requires the original artifact, not an edited protocol copy.

## Historical Hashes and Code Versions

Historical SHA256 values are retained in provenance JSON. They describe the original execution input/code/protocol/checkpoint, not the sanitized public file. [BUILD_MANIFEST.json](BUILD_MANIFEST.json) records each selected artifact's source identity as root-environment name plus relative path components, source hash, initial-copy hash, final public hash, and sanitization status. No personal source prefix is required to interpret that mapping.

The audited source HEAD is `c21f2ae314351832fba924674a7406a910de700a`. PPO's recorded execution commit is `34b1738a857745d22ed37bea17de8d4065a014c8`. The recorded `orchestrate.py`, `summarize.py`, and `train_target.py` hashes match the latter Git blobs, while the copied source files include later recovery edits. Public files additionally include portable routing changes. The [execution-code hashes](privileged_ppo/EXECUTION_CODE_HASHES.json), [run manifest](privileged_ppo/RUN_MANIFEST.json), and [historical final verification](privileged_ppo/FINAL_VERIFICATION.json) retain that distinction.

Checkpoint/optimizer/RNG recovery did not restore complete PhysX internal state. It is not exact replay. Missing original source-training seed logs prevent certification of historical training-seed overlap.

## Scientific Interpretation

- October mass8 evaluations use the v2 reset/expert-warmup and single-episode contract. Historical September PPO comparisons use a different reset/controller contract. Their returns must not be pooled.
- The initial 20-seed observation that 30k underperformed 10k was revised by the larger 100-seed sample. Keep the sample-size lineage explicit.
- Candidate diagnostics include command-path and simulator-state restoration limitations; they are not pure f/R/V causal intervention results.
- Adaptation-chain variance includes initialization/config/split differences; shared evaluation seeds are not independent training repeats.
- T1 equal-total composition and T3 support n5 are inconclusive. T2 failed its protocol/reset gate. The incomplete T1 downstream-control family and failed T3A n10 are not promoted to completed evidence.
- The archived module and chain reports also describe auxiliary friction/gain/flat work. The public selected seed tables emphasize the mass8 stories; full auxiliary raw evidence is external.

## Static Validation

Run `python -B tools/verify_portfolio.py` to check syntax/compilation, selected file presence/hashes, local import targets, internal README links, selected CSV readability, PNG structure, and private-path/credential patterns. This validator does not import experiment modules, start IsaacSim, load checkpoints, or run experiments.

[VALIDATION.json](VALIDATION.json) records the creation-time checks, including source-integrity and scientific-value preservation checks against the original files. `STATIC_PASS` means static checks passed. External package/runtime availability is a separately declared requirement; no substitute modules were created.
