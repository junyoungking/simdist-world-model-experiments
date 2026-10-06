# T1 data-amount / buffer-composition control report

## Result

Final label: **`INCONCLUSIVE_DATA_COMPOSITION`**. The primary contrast is Δ_comp = L_mixed − L_current_only; negative values favor replacing 500 extra +8 kg windows with 500 historical +4 kg windows.

## Required verification

1. Yes. Both arms in every repeat independently loaded the identical M_pre source checkpoint at absolute step 367004.
2. Yes. `B_core` is exactly the same 500 window IDs in both arm manifests.
3. Yes. `B_extra` and `A_hist` contain exactly 500 unique windows each.
4. Yes. Each arm's explicit post-processing optimizer-eligible training manifest contains exactly 1,000 unique windows.
5. Yes. B_eval is a separate immutable +8 kg `role=eval` artifact and all per-repeat leakage checks pass.
6. Yes. Every arm has K=1,000 actual relative optimizer updates (367004→368004), verified from the Trainer completion counter and checkpoint audit.

## Raw repeat results

| Repeat | L_pre | L_current_only | L_mixed | Δ_comp |
|---:|---:|---:|---:|
| 1 | 0.044592780 | 0.033542096 | 0.033192278 | -0.000349818 |
| 2 | 0.031918046 | 0.027195175 | 0.026610838 | -0.000584337 |
| 3 | 0.043531800 | 0.032082953 | 0.031147347 | -0.000935606 |
| 4 | 0.040397008 | 0.029487026 | 0.029301421 | -0.000185605 |
| 5 | 0.038050813 | 0.027512683 | 0.025452955 | -0.002059728 |

The five raw L_current_only, L_mixed and Δ_comp values are also recorded without aggregation in `../03_results/DATA_AMOUNT_CONTROL_RAW.csv`.

## Repeat-level statistics

Δ_comp: mean -0.000823019; median -0.000584337; sample SD 0.000746544; SE 0.000333864; 95% t CI [-0.001749975, 0.000103938]; signs positive/negative = 0/5; min/max [-0.002059728, -0.000185605].

## Adaptation relative to M_pre

- r1: CURRENT_ONLY change -0.011050684 (-24.78%); MIXED change -0.011400502 (-25.57%).
- r2: CURRENT_ONLY change -0.004722871 (-14.80%); MIXED change -0.005307208 (-16.63%).
- r3: CURRENT_ONLY change -0.011448847 (-26.30%); MIXED change -0.012384453 (-28.45%).
- r4: CURRENT_ONLY change -0.010909982 (-27.01%); MIXED change -0.011095587 (-27.47%).
- r5: CURRENT_ONLY change -0.010538130 (-27.69%); MIXED change -0.012597858 (-33.11%).

Both arms improved in every repeat: CURRENT_ONLY=True; MIXED=True.

## Relation to prior natural-buffer T1

The immutable prior E1 CUMULATIVE-NATURAL minus RECENT values were -0.003122888, -0.001509063, -0.004621414, -0.002789333, -0.004510154 (all negative; mean -0.003310570). That experiment had natural buffer growth and unequal current-regime exposure, so it is not overwritten here.
Combined interpretation: **INCONCLUSIVE_DATA_COMPOSITION**. The previous advantage cannot yet be attributed to historical-regime contribution versus data amount.

## Remaining evidence for the research-anchor decision

This narrow prediction-only gate supplies no control-return, tracking, planner, R/V, or support-coverage evidence. Because the composition CI still spans zero, the specific missing evidence is a pre-registered repeat-level precision replication of this same fixed-size contrast sufficient to distinguish the sign from repeat variability; no new hypothesis suite is justified by this run. The anchor decision should use that evidence together with the immutable prior T1 record.
