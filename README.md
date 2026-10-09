# SimDist 기반 World Model 적응 및 제어 실험

SimDist를 기반으로 dynamics가 바뀐 환경에서 World Model을 적응시키고, prediction 성능이 실제 MPPI 제어로 이어지는지 실험했습니다. 모든 결과는 IsaacLab의 Go2 시뮬레이션에서 수행한 **sim-to-sim 실험**입니다.

처음에는 adaptation에 사용할 데이터의 구성을 비교했습니다. 그런데 예상과 반대의 결과가 나왔고, 통제 조건을 바꾸면 차이도 크게 달라졌습니다. Prediction loss만 보고 결론을 내리기 어렵다고 판단해 MPPI control을 직접 평가했고, 이후에는 planner의 candidate ranking과 adaptation chain 사이의 차이를 분석했습니다. 현재는 friction과 joint gain 변화로도 실험을 확장하고 있으며, 진행 중인 결과는 7절에 따로 정리했습니다.

World Model 구조, 기본 MPPI controller, 사전 학습 checkpoint와 시뮬레이터 통합 코드는 [원본 SimDist](https://github.com/CLeARoboticsLab/simdist)를 사용했습니다. 이 repository에는 제가 추가한 실험 및 분석 코드와 공개 가능한 결과 일부를 모았습니다.

## 1. Adaptation 데이터 구성 실험 (T1~T3)

### T1: 과거 데이터 누적

Dynamics가 바뀌면 이전 regime의 데이터가 adaptation을 방해할 수 있다고 생각했습니다. 그런데 +4 kg의 과거 데이터를 누적한 쪽이 recent-only보다 5번 모두 prediction loss가 낮았습니다. Abrupt +4 → +8 kg 조건에서 `CUM - REC`의 평균 차이는 `-0.00331`이었습니다.

이 비교에서는 과거 데이터를 섞은 쪽의 데이터 양도 더 많았습니다. 양쪽을 각 1,000 window로 맞춘 별도 실험에서는 과거 데이터를 섞은 쪽과 recent-only의 차이가 `-0.00082`로 줄었고, 95% CI는 `[-0.00175, +0.00010]`이었습니다. 두 실험은 버퍼 구성 방식도 달라 데이터 양의 효과만 분리했다고 볼 수는 없습니다. 처음 관찰을 **과거 데이터 자체의 효과로 보기 어려웠다**는 것이 현재의 결론입니다.

### T2: Planner 순위 검증과 synchronization

같은 초기 state에서 여러 action sequence를 실행한 실제 return으로 planner의 예측 순위를 검증하려 했습니다. 그러나 병렬 환경의 reset과 synchronization 조건이 맞지 않았습니다. 동일한 action prefix를 적용한 후에도 state error가 최대 약 `3.3`으로 허용치 `1e-5`를 넘어서, 유효한 비교를 만들 수 없었습니다.

이 실패 이후에는 3절에서처럼 진행 중인 시뮬레이션의 state를 저장하고 복원해 candidate를 실행하는 별도 진단을 구성했습니다. 그 방식도 PhysX 내부 state를 완전히 복원하지 못하므로 엄밀한 causal 비교와는 구분했습니다.

### T3: 데이터 support

평가 state에 가까운 데이터를 선택하면 adaptation에 더 도움이 될 것으로 예상했습니다. 각 command의 evaluation state를 기준으로 latent+action 공간에서 가까운 데이터 500개(MATCHED)와 먼 데이터 500개(DISTANT)를 골라 학습했습니다. n=5 반복에서 command별 차이의 confidence interval은 모두 0을 포함했고 방향도 일정하지 않았습니다. 가까운 데이터를 쓰는 것이 일관되게 유리하다는 결론은 얻지 못했습니다.

실험 근거: [T1](results/t1/DATA_AMOUNT_CONTROL_SUMMARY.csv), [T2 synchronization](results/t2/T2V2_SYNC_GATE.csv), [T3](results/t3/FRESH_REPLICATION_SUMMARY.csv)

## 2. Prediction error와 control 성능

Go2의 base mass를 +8 kg으로 바꾼 target 환경에서 데이터를 수집하고 dynamics component `f`만 적응시켰습니다. Encoder, reward, value, policy component는 고정했습니다.

먼저 하나의 adaptation chain을 100개 evaluation seed로 평가했습니다. MSE는 학습에 사용하지 않은 **M0-policy 독립 검증 세트(10 seed × 500 step)의 seed별 25-step latent MSE 평균**입니다.

| Model | 25-step latent MSE | Mean return | 실패 / 100 |
|---|---:|---:|---:|
| M0 | 0.064 | 16.686 | 62 |
| 10k | 0.052 | 22.951 | 50 |
| 30k | 0.046 | 25.123 | 40 |

이 chain에서는 error와 실패 횟수가 함께 줄었습니다. 처음 20-seed 평가에서는 30k가 10k보다 나쁜 결과도 있었지만, 평가를 100 seed로 늘리자 순서가 바뀌었습니다.

이어 다른 데이터 구성으로 이어 학습한 30k_c → 60k → 90k를 비교했습니다. 아래 MSE도 첫 번째 표와 **같은 M0-policy 독립 검증 세트(seed 35001~35010)**에서 seed별로 평균한 값입니다. 다만 첫 번째 30k와 30k_c는 데이터 구성과 학습 이력이 다른 checkpoint입니다.

| Stage | 25-step latent MSE | 실패 / 100 |
|---|---:|---:|
| 30k_c | 0.0450 | 40 |
| 60k | 0.0419 | 44 |
| 90k | 0.0408 | 37 |

Prediction error는 약 9% 감소했지만 같은 100개 평가 seed에서 control failure가 뚜렷하게 좋아졌다고 보기는 어려웠습니다. 100개 seed는 **100개 독립 학습 모델이 아니라 동일한 모델을 평가한 100개 episode**입니다.

코드: [데이터 수집](experiments/adaptation_curve_v1/collect.py), [Adaptation](experiments/adaptation_curve_v1/train_stage.py), [Control evaluation](experiments/score_probe_v1/eval_probe_batch.py)

결과: [초기 chain 평가](results/score_probe_v1/analysis/a_conditions.csv), [초기 chain loss](results/score_probe_v1/analysis/c_loss_summary.csv), [연장 평가](results/extended_adaptation_v1/aggregate/condition_summary.csv), [연장 loss](results/extended_adaptation_v1/aggregate/independent_loss_clean.csv)

## 3. Counterfactual rollout으로 본 candidate ranking

MPPI는 여러 action sequence의 미래 return을 예측해 상대적으로 높은 후보를 선택합니다. 2절의 prediction 개선이 실제로 좋은 action의 선택으로 이어지는지 보려고 저장한 state에서 여러 candidate를 실행하고 예측과 결과를 비교했습니다.

초기 실험에서는 **371개 planning snapshot, 25,228개 candidate branch**를 분석했고 그중 24,424개가 25-step을 끝까지 실행했습니다. Adaptation으로 25-step latent prediction error가 약 28% 줄었어도 candidate ranking의 Spearman correlation은 대략 `0.70 → 0.72`로 거의 그대로였습니다. 실제 성능 상위 8개 후보 중 모델이 상위 8개로 평가한 후보는 평균 약 2개였습니다.

Expert action도 눈에 띄었습니다. 실제 latent에 같은 head를 적용하면 상위권(조건에 따라 평균 약 3~7등)이었지만 predicted latent 기준 순위는 M0에서 약 50등, 적응 후에도 약 33등이었습니다. 모델이 전체 latent를 더 잘 예측해도 **planner가 선택해야 할 후보의 상대적 평가가 충분히 개선되지 않을 수 있다**는 사례였습니다.

다만 저장한 상태로 복원할 때 PhysX 내부 state를 완전히 재현하지는 못했고, 예측 시 가정한 command와 실제 실행 command도 일부 달랐습니다. 따라서 이 비교를 `f`, reward, value의 순수한 causal contribution으로 해석하지는 않았습니다.

코드: [Candidate 실행](experiments/candidate_rank_v1/run_candidates.py), [State 복원](experiments/candidate_rank_v1/state.py), [Ranking 분석](analysis/candidate_rank_v1/analyze.py)

결과: [Ranking summary](results/candidate_rank_v1/analysis/ranking_summary.csv), [Expert rank](results/candidate_rank_v1/decomposition_v1/expert_rank_summary.csv), [진단 한계](docs/candidate_decomposition.md)

## 4. Adaptation chain 간 차이

Adaptation을 다시 시작해 별도로 데이터 수집과 학습을 반복했습니다. 동일한 +8 kg target에서 **원래 policy head(P0)를 사용하는 MPPI로 데이터를 수집한 네 chain**을 비교한 결과입니다. 100-seed와 200-seed 실패율은 전체 episode 기준(최대 1,000 step, 넘어지면 종료)입니다. 200-seed 평가는 기존 평가 seed 100개에 새 seed 100개를 추가했습니다. 아래 MSE는 2절과 **다른 독립 검증 세트**에서 계산했으므로 같은 checkpoint라도 값이 다릅니다.

| Chain | 25-step latent MSE | 실패 / 100 | 실패율 / 200 |
|---|---:|---:|---:|
| M0 | 0.0642 | 62 | 64.5% |
| Chain 1 | 0.0442 | 40 | 43.5% |
| Chain 2 | 0.0375 | 49 | 48.0% |
| Chain 3 | 0.0352 | 37 | 40.0% |
| Chain 4 | 0.0379 | 52 | 58.5% |

100-seed에서 Chain 1과 Chain 2의 실패 수는 각각 40회와 49회였습니다. 평가를 200 seed로 늘려도 prediction error가 네 chain 중 가장 높은 Chain 1이 Chain 4보다 15%p 낮은 실패율을 보였습니다. 전체 여섯 adaptation chain(P0 수집 네 개와 변형된 policy로 수집한 두 개)의 같은 200-seed 평가 결과는 원자료에서 재집계했습니다. 다만 과거 집단 검정의 원래 통계량과 exact p 산출 경로는 찾지 못해 `NOT_REPRODUCED`로 표시하고 검정 결론은 제외했습니다. 같은 evaluation seed와 별도로 학습한 chain을 구분하며, 대체 검정은 실시하지 않았습니다.

실패율 변화가 evaluation seed의 변동과 adaptation chain의 차이 중 어디에서 왔는지 구분하기 위해 100 seed 한 번의 순위에 의존하지 않았습니다. 이 표는 공통 검증 세트에서 loss가 낮은 순서와 제어 실패율 순서가 반드시 일치하지 않는다는 점을 보여줍니다.

결과: [Chain comparison](results/positive_control_variance_v1/mass8/aggregate/chain_comparison.csv), [Control summary](results/positive_control_variance_v1/mass8/aggregate/condition_summary.csv), [Independent loss](results/positive_control_variance_v1/mass8/losses_summary.csv), [200-seed 집계와 수집 조건](results/portfolio_evidence_20261009/mass_chains_200seed.csv), [6-chain 검정 재현 상태](results/portfolio_evidence_20261009/chain_equality_test.csv).

## 5. Dynamics와 policy 조합

Dynamics와 policy head를 따로 적응시켜 조합했습니다. `D30`은 첫 chain의 30k dynamics checkpoint이고, `P50`은 +8 kg target의 **privileged expert 데이터 10k step**으로 policy head만 50 update 학습한 진단용 모델입니다. P50의 숫자 50은 데이터 양이 아니라 optimizer update 수입니다.

| 구성 | Mean return |
|---|---:|
| M0-P0 | 16.686 |
| D30-P0 | 25.123 |
| M0-P50 | 24.049 |
| D30-P50 | 19.148 |

D30과 P50은 각각 baseline보다 좋았지만 조합한 모델은 두 단독 적용보다 낮았습니다(원래 M0-P0보다는 높음). Paired 100 seed에서 `D30-P50 - D30-P0 ≈ -6.0`이고 95% CI는 약 `[-9.4, -2.6]`입니다. D90으로 바꿔도 차이는 약 `-6.4`였습니다.

한 모델 조합에서 나타난 차이이며, 4절의 adaptation chain 간 편차도 무시하기 어렵습니다. 두 component가 개별적으로 좋아졌을 때 항상 함께 좋아지는지에 대한 일반적인 결론은 내리지 않았습니다.

코드: [Policy 학습](experiments/module_combo_v1/train_policy_50.py), [Checkpoint 구성](experiments/module_combo_v1/compose_policy_checkpoint.py), [분석](analysis/module_combo_v1/aggregate.py)

결과: [Module comparison](results/module_combo_v1/mass8/aggregate/condition_summary.csv), [Paired comparison](results/module_combo_v1/mass8/aggregate/paired_summary.csv)

## 6. Privileged PPO reference

+8 kg target 자체가 제어 불가능한 환경인지 구분하기 위해 원본 SimDist에서 사전 학습한 **privileged PPO expert**를 동일한 v2 평가 규약으로 실행했습니다. 아래 표는 별도의 20-seed 평가 결과로, 2절의 100-seed 평균과 표본 수가 다릅니다. PPO는 privileged observation을 사용하므로 World Model + MPPI와 공정한 알고리즘 대결로 볼 수 없습니다.

| 환경과 controller | 실패 / 20 | Mean return |
|---|---:|---:|
| Source, M0 + MPPI | 1 | 36.45 |
| Mass +8 kg, M0 + MPPI | 14 | 14.21 |
| Mass +8 kg, PPO expert | 0 | 45.80 |

같은 +8 kg target에서 PPO expert는 20개 episode를 모두 완주했습니다. 따라서 task 자체는 수행 가능하지만, pretrained World Model + MPPI가 target dynamics에서 충분한 제어 성능을 내지는 못한다는 기준점을 얻었습니다. Privileged state를 쓰는 PPO의 숫자를 현실에 그대로 적용할 수는 없습니다.

코드: [초기 평가](experiments/eval_s1_v2_batch.py), [100-seed 비교](experiments/score_probe_v1/eval_probe_batch.py)
결과: [Adaptation curve aggregate](results/adaptation_curve_v1/aggregate/condition_summary.csv)

## 7. 다른 target으로 확장 (2026년 10월 기준, 진행 중)

Mass +8 kg에서 시작한 분석을 friction과 joint gain 변화, 낮은 마찰의 평지 환경으로 확장했습니다. 현재 결과에서는 **문제가 두드러지는 부분이 target마다 달랐습니다**. Mass에서는 dynamics 적응이 도움이 되는 경우가 많았고, friction에서는 dynamics prediction과 value ranking을 함께 살펴봐야 했으며, gain에서는 value ranking의 변화가 특히 컸습니다. 이는 관찰된 병목의 후보이지, 각 모듈을 개입해 원인을 완전히 분리한 결론은 아닙니다.

### 평가 규약

IsaacLab Go2에서 25-step PPO expert warm-up 후 seed당 최대 1,000 step을 실행하고, 넘어지면 종료했습니다. 이 절의 제어 평가는 **200 seed**(23501~23600 및 24001~24100)를 기준으로 합니다. Boxes 지형의 `tile 안 실패율`은 전진 변위가 7.425 m를 넘기 전에 넘어진 비율이고, mass, friction, gain은 1.0 m/s에서 평가했습니다. 평지 friction 0.10은 0.75 m/s에서 평가한 전체 episode 실패율로, tile 지표가 아닙니다. 4절의 mass chain을 200 seed에서 **tile 안 실패만 다시 집계한 것**이므로 4절의 전체 episode 실패율과 숫자가 다릅니다.

### Value head 순위와 dynamics 구간 오차

Sim에서 학습한 value head의 순위가 target에서도 유지되는지 확인하기 위해, 저장한 state에서 PPO expert를 실행해 얻은 Monte Carlo return과 value 예측의 Spearman 상관을 비교했습니다. 아래 표는 10월 7일 기준 실험이며, value head와 privileged critic의 관측 조건이 같지는 않습니다. MC horizon은 25-step 뒤 continuation 기준 source/mass에서 175 step, friction/gain에서 200 step입니다. 각 target 내부의 두 상관계수는 같은 MC와 snapshot 쌍으로 계산했지만, 조건 간 MC return을 직접 비교하거나 동일 horizon으로 해석하지 않았습니다.

| 환경 | Value head 대 MC | Privileged critic 대 MC |
|---|---:|---:|
| Source | 0.57 | 0.76 |
| Mass +8 kg | 0.57 | 0.77 |
| Friction 0.15 | 0.39 | 0.68 |
| Gain 0.7배 | 0.05 | 0.44 |

Mass에서 value head의 순위 상관은 source와 같았지만, friction에서는 낮아졌고 gain에서는 0.05까지 떨어졌습니다. Gain은 privileged critic도 0.44로 낮아졌기 때문에 value head에만 생긴 현상으로 볼 수 없습니다.

다음은 **같은 target 안에서 M0와 adapted checkpoint를 공통 state에 놓고 비교한** reward prediction 오차입니다. Dynamics 구간 오차는 predicted latent를 reward head에 넣은 점수와 실제 rollout latent를 같은 reward head에 넣은 점수의 차이이며, 미래 command path 차이도 일부 포함합니다.

| Target | M0 오차 | Adapted 오차 |
|---|---:|---:|
| Mass +8 kg | 0.152 | 0.100~0.111 |
| Friction 0.15 | 0.308 | 0.255~0.271 |
| Gain 0.7배 | 0.171 | 0.144~0.146 |
| 평지 friction 0.10 | 0.345 | 0.230~0.256 |

공통 state에서는 **네 target 모두 adaptation 후 오차가 줄었습니다**. 감소 후에도 잔여 오차는 mass가 가장 작았고, friction과 평지는 상대적으로 컸습니다. 평지 행의 adapted checkpoint는 **1.0 m/s 데이터로 적응한 모델**이고, 공통 state의 평가 속도는 **0.75 m/s**입니다. 이 모델은 공통 state의 dynamics 구간 오차가 `0.345 → 0.230~0.256`으로 줄었지만, 같은 0.75 m/s 환경에서 전체 episode 실패율은 M0의 `53%`에서 `91.5~94.5%`로 증가했습니다. 예측 오차가 감소해도 제어 개선으로 연결되지 않은 사례입니다.

### Prediction loss와 실패율

아래 loss 감소율은 각 target의 held-out 평가 요약을 기준으로 했습니다. Target held-out은 186001~186020의 seed별 MSE 평균(n=20)이며, mass의 감소율은 32.299~46.356%입니다. 4절은 별도 독립 수집 seed 49301~49310의 MSE 평균(n=10)으로, window 수로 가중한 pooled 평균과 구분했습니다. Mass는 약 32~46%이며, 4절에 사용한 **별도 독립 검증 세트**의 `0.0642 → 0.0352~0.0442`로 계산하면 약 31~45%입니다. 실패율의 기준은 **mass, friction, gain 모두 boxes tile 안**, 평지는 0.75 m/s의 전체 episode입니다. 7절의 mass 실패율은 4절의 mass chain을 같은 200 seed에서 tile 안 실패만 다시 집계한 값입니다.

| Target | 25-step loss 감소 | M0 실패율 | Dynamics 적응 후 | 해석 |
|---|---:|---:|---|---|
| Mass +8 kg | 약 32~46% (4절 검증 세트: 31~45%) | 62.5% | 38, 41.5, 43.5, 46.5, 57, 58% | 회차별 감소폭 차이 |
| Friction 0.15 | 약 42~46% | 81.5% | 74, 74.5, 79, 87.5% | 회차별 실패율 차이 |
| Gain 0.7배 | 약 43~45% | 58% | 43%, 64.5% | 2개 chain 중 1개 개선 |
| 평지 friction 0.10 (0.75 m/s 재적응) | 약 84~85% | 53% | 58%, 62% | 일관된 개선 확인 안 됨 |

평지에서 **0.75 m/s 데이터로 다시 적응한 두 모델**은 같은 속도에서 실패율이 58%, 62%로 나타나 큰 악화가 사라졌지만, M0 대비 개선까지 확인되지는 않았습니다. Friction target에서는 boxes tile 밖의 저마찰 평지 구간에서 모델 간 차이가 더 크게 나타났습니다. 따라서 loss와 실패율을 해석할 때 데이터 수집 속도, 실제 평가 속도, 지형 구간을 분리해야 했습니다.

### 진행 중인 실험

Dynamics 외에 encoder, reward, value까지 적응시키는 비교도 진행 중입니다. 현재는 학습 loss의 상대적 크기와 value target 구성의 영향을 분리하는 단계여서 성능 결론을 포함하지 않았습니다.

이 절의 집계 요약: [Value/critic 순위](results/portfolio_evidence_20261009/value_mc_rank.csv), [공통 state 오차](results/portfolio_evidence_20261009/common_state_dynamics_error.csv), [200-seed target 실패율](results/portfolio_evidence_20261009/target_failure_summary.csv), [속도 일치 대조](results/portfolio_evidence_20261009/flat_velocity_comparison.csv), [Mass chain 200-seed 집계](results/portfolio_evidence_20261009/mass_chains_200seed.csv). 원자료 재집계 여부와 분석 범위는 [근거 파일 안내](results/portfolio_evidence_20261009/README.md)에 적었습니다.

## Reproducibility

원본 구현은 [CLeARoboticsLab/SimDist](https://github.com/CLeARoboticsLab/simdist)이며 기준 버전은 SimDist `b63f76ba01457b4df0954f88985a3a73a206ec17`, IsaacLab `157c6b74ed4d9892c5e5ccc0d38d7835f27e98f9`(2.0.1)입니다.

이 repository에는 연구 코드, 일부 CSV, 문서와 진단용 instrumentation patch를 포함했습니다. 대용량 checkpoint와 전체 rollout 데이터는 포함되지 않습니다. Static verification은 코드 의존성 및 경로를 확인하는 것이며 학습과 시뮬레이션 재실행을 뜻하지 않습니다.

- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Raw evidence builder](analysis/build_portfolio_evidence_from_raw.py), [집계 규칙과 공개 범위](analysis/PORTFOLIO_RAW_PROVENANCE.md)
- [MPPI patch](patches/mppi.patch)
- [Trainer patch](patches/trainer.patch)
- [Original SimDist license](docs/licenses/SimDist_LICENSE.txt)
