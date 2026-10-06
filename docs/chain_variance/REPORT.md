# 평지 양성 대조군과 +8kg dynamics 체인 편차

초기 예상 5–6시간, 평지 실측 반영 후 7–8시간. 전체 예상이 9시간 이하여서 2번도 실행했다. 실제 소요 약 6.41시간.

모든 새 코드와 결과는 원 연구 root에 작성했다. 공개본은 experiments/positive_control_variance_v1과 results/positive_control_variance_v1로 분리했다. 기존 파일의 SHA256 불변 확인을 통과했다.

## 규약 및 지형

v2: 기본 관절 위치·속도 reset, PPO expert 25-step warm-up, 목표 주행 최대 1,000 step (50 Hz). 실패는 평가 terminated 또는 warm-up 실패다. 표의 ±는 seed 간 표본 SD다. 목표 속도는 기존 경로 추종 command의 forward_vel이며 제어 규약은 그대로다.

평지는 IsaacLab TerrainImporterCfg의 terrain_type='plane', terrain_generator=None인 무한 평면이다. height scanner와 433-dim expert 관측, 35/315-dim 모델 관측 및 reward는 유지했다. 초기 ray hit world-z span <1e-5 m를 매 seed 확인했다. 질량 +0kg, Kp/Kd=25/0.5, 발 restitution=0.05. +8kg은 기존 boxes difficulty=0.4, 마찰 0.8, 목표 1.0m/s다.

## 1. 학습 전 평지 조건 탐색 (23501–23520)

| 발 마찰 | 목표 m/s | m0 MPPI 실패 | expert 실패 | 적격 | 범위까지 거리 |
|---:|---:|---:|---:|---|---:|
| 0.15 | 0.5 | 0/20 | 0/20 | False | 8 |
| 0.15 | 1.0 | 2/20 | 0/20 | False | 6 |
| 0.80 | 0.5 | 0/20 | 0/20 | False | 8 |
| 0.80 | 1.0 | 0/20 | 0/20 | False | 8 |
| 0.10 | 0.5 | 2/20 | 0/20 | False | 6 |
| 0.10 | 1.0 | 20/20 | 0/20 | False | 3 |

선택: 마찰 0.10, 목표 1.0m/s. 이유: 적격 조건이 없어 expert 실패≤1 및 m0 실패 8–17까지의 실패 수 거리 합이 최소인 조건을 선택했다.
거리 = max(expert 실패−1,0) + max(8−m0 실패,m0 실패−17,0). 동률이면 마찰 0.15/0.5m/s 우선, 이후 expert 실패 수, m0 실패 수의 구간 중앙(12.5)까지 거리, 조건명으로 정했다.

## 평지 100-seed 평가 (23501–23600)

| 모델 | 실패 | 길이 | return | reward/step |
|---|---:|---:|---:|---:|
| m0 | 100/100 | 140.3 ± 139.5 | 3.832 ± 4.460 | 0.0256 ± 0.0038 |
| chain1 | 100/100 | 286.4 ± 169.2 | 6.396 ± 4.392 | 0.0211 ± 0.0062 |
| chain2 | 100/100 | 233.9 ± 174.9 | 5.251 ± 5.353 | 0.0200 ± 0.0077 |
| base | 100/100 | 532.0 ± 0.0 | 19.336 ± 0.000 | 0.0363 ± 0.0000 |

| 비교 (왼쪽−오른쪽) | Δ 실패율 | Δ 길이 | Δ return | Δ reward/step | 실패 교차: 둘/왼쪽만/오른쪽만/생존 |
|---|---:|---:|---:|---:|---:|
| chain1-minus-m0 | 0.000 ± 0.000 | 146.1 ± 229.1 | 2.564 ± 6.663 | -0.0045 ± 0.0079 | 100/0/0/0 |
| chain2-minus-m0 | 0.000 ± 0.000 | 93.6 ± 218.9 | 1.419 ± 6.761 | -0.0056 ± 0.0088 | 100/0/0/0 |
| base-minus-m0 | 0.000 ± 0.000 | 391.7 ± 139.5 | 15.504 ± 4.460 | 0.0107 ± 0.0038 | 100/0/0/0 |
| chain1-minus-chain2 | 0.000 ± 0.000 | 52.5 ± 234.0 | 1.145 ± 6.797 | 0.0011 ± 0.0090 | 100/0/0/0 |
| chain1-minus-base | 0.000 ± 0.000 | -245.6 ± 169.2 | -12.940 ± 4.392 | -0.0152 ± 0.0062 | 100/0/0/0 |
| chain2-minus-base | 0.000 ± 0.000 | -298.1 ± 174.9 | -14.085 ± 5.353 | -0.0163 ± 0.0077 | 100/0/0/0 |

독립 검증: m0 MPPI 10 seed (45301–45310) × 500 control step. 학습·중단 기준에 사용하지 않았다.

| 모델 | 1-step latent MSE | 25-step latent MSE |
|---|---:|---:|
| m0 | 0.03597 ± 0.00777 | 0.09918 ± 0.03589 |
| chain1 | 0.00827 ± 0.00370 | 0.01288 ± 0.00586 |
| chain2 | 0.00811 ± 0.00390 | 0.01386 ± 0.00925 |

평지 base policy는 100개 seed에서 모두 길이 532 step이었다. 평면·고정 물성과 deterministic head에서 같은 결과가 반복됐으므로 이 seed 반복을 서로 다른 지형 표본으로 해석하지 않는다.

평지 두 체인은 43101–43106/43201–43206, 44101–44106/44201–44206 수집; 학습 seed 43501/43502, 44501/44502. 단계별 6개 수집 seed 중 마지막은 held-out이며 train/held-out을 단계 간 누적했다.

## 2. +8kg 체인별 수집·loss·실패

평가 시 모든 dynamics에 원래 policy P0를 사용했다. D_P 계열은 P50를 고정하여 수집·학습하고 평가용 checkpoint에는 P0를 이식했다. 수집 총량은 체인당 30,000 control step. warm-up은 수집 step과 window에서 제외했다.

| 체인 | 수집 정책 | 넘어짐 | warm-up 넘어짐 | 근접 window / 전체 (%) | 독립 25-step MSE | 실패 |
|---|---|---:|---:|---:|---:|---:|
| old_chain1 | MPPI (D_stage, original P0) | 42 | 0 | 2991/26740 (11.19%) | 0.04421 ± 0.01988 | 40/100 |
| old_chain2 | MPPI (D_stage, original P0) | 47 | 0 | 3435/26385 (13.02%) | 0.03749 ± 0.01345 | 49/100 |
| old_DP | MPPI (D_stage, P50) | 20 | 0 | 1483/27500 (5.39%) | 0.03810 ± 0.01075 | 62/100 |
| new_chain3 | MPPI (D_stage, original P0) | 39 | 0 | 2720/26617 (10.22%) | 0.03515 ± 0.01350 | 37/100 |
| new_chain4 | MPPI (D_stage, original P0) | 39 | 0 | 2458/26566 (9.25%) | 0.03793 ± 0.01490 | 52/100 |
| new_DP2 | MPPI (D_stage, P50) | 25 | 0 | 1847/27253 (6.78%) | 0.03844 ± 0.01163 | 37/100 |

m0: 실패 62/100, 독립 25-step MSE 0.06420 ± 0.01868.

근접 window: L개 control step인 episode에서 유효 window start=0..L−51, 예측 anchor t=start+25. terminated episode에 대해 1≤L−t≤100인 window 수 / 모든 유효 window 수. 동일 window 규칙으로 체인 간 계산했다. overlap 기준과 control-step 기준 비율도 collection CSV에 보조로 저장했다. timeout과 budget-censored의 끝을 넘어짐으로 세지 않았다.

독립 세트는 새 m0 MPPI seed 49301–49310 × 500 step (mass8/validation)이며, 모든 모델을 동일 episode/window에 재평가했다. 기존 m0 검증 세트의 seed 35001은 첫 체인의 후속 30k_c 분기에서 cp10000 중단용 자료와 simulator seed가 같아 사용하지 않고 새 세트를 수집했다.

old_chain1은 기존 첫 30k checkpoint (stage_30000, step 380304)와 기존 4개 1k/2k/7k/20k 수집 파일; old_chain2는 followup_v1의 두 번째 체인; old_DP는 module_combo_v1/followup_repro/mass8_d_p다. 첫 체인은 기존 수집 단계·split이 새 체인과 다르므로 완전히 대칭적인 반복 실험은 아니다.

새 P0 체인 수집 seed: 46101–46106/46201–46206 및 47101–47106/47201–47206; 학습 46501/46502 및 47501/47502. 새 D_P: 48101–48106/48201–48206; 학습 48501/48502.

## +8kg m0 대비 같은 seed 차이 및 실패 교차표

| 비교 (왼쪽−오른쪽) | Δ 실패율 | Δ 길이 | Δ return | Δ reward/step | 실패 교차: 둘/왼쪽만/오른쪽만/생존 |
|---|---:|---:|---:|---:|---:|
| old_chain1-minus-m0 | -0.220 ± 0.613 | 191.6 ± 465.8 | 8.437 ± 17.379 | 0.0028 ± 0.0071 | 30/10/32/28 |
| old_chain2-minus-m0 | -0.130 ± 0.614 | 126.8 ± 464.0 | 5.928 ± 17.171 | 0.0027 ± 0.0066 | 36/13/26/25 |
| old_DP-minus-m0 | 0.000 ± 0.711 | 34.1 ± 541.4 | 1.780 ± 19.776 | 0.0007 ± 0.0073 | 37/25/25/13 |
| new_chain3-minus-m0 | -0.250 ± 0.609 | 215.6 ± 453.4 | 9.301 ± 16.754 | 0.0034 ± 0.0064 | 28/9/34/29 |
| new_chain4-minus-m0 | -0.100 ± 0.659 | 95.4 ± 497.7 | 3.761 ± 17.981 | 0.0010 ± 0.0070 | 35/17/27/21 |
| new_DP2-minus-m0 | -0.250 ± 0.657 | 210.9 ± 490.2 | 7.726 ± 17.495 | 0.0018 ± 0.0070 | 25/12/37/26 |

전체 체인 간 짝 비교는 mass8/aggregate/paired_summary.csv 및 paired_seed_differences.csv에 있다.

## 학습 규칙 및 원본

dynamics-only latent loss, batch 32, 기존 warm-up cosine Adam, 단계마다 fresh optimizer. 10 epoch update cap; 100-update 간격 held-out 검사; 최소 500 update; 상대 0.1% 개선 없는 검사 5회 시 중단; best checkpoint 선택. 모든 새 학습에서 frozen parameter 변경 0 확인.

| 체인/단계 | train/held-out window | 선택/실행 update | held-out loss |
|---|---:|---:|---:|
| flat/chain1/10k | 4252/643 | 1320/1320 | 0.02227 |
| flat/chain1/30k | 16688/3326 | 5200/5210 | 0.00943 |
| flat/chain2/10k | 4757/477 | 1480/1480 | 0.01986 |
| flat/chain2/30k | 18085/3055 | 5650/5650 | 0.01094 |
| mass8/new_DP2/10k | 7184/1566 | 2200/2240 | 0.02336 |
| mass8/new_DP2/30k | 22554/4699 | 3300/3800 | 0.01861 |
| mass8/new_chain3/10k | 7370/1269 | 2100/2300 | 0.04782 |
| mass8/new_chain3/30k | 22365/4252 | 2700/3200 | 0.03380 |
| mass8/new_chain4/10k | 6884/1116 | 2000/2150 | 0.03899 |
| mass8/new_chain4/30k | 22367/4199 | 1900/2400 | 0.02341 |

원본 root: `artifact://positive_control_variance_v1/results`.

- screen_seed_results.csv 및 screen/: 6조건 × 2제어기 × 20 seed summary.json, warmup_steps.csv, evaluation_steps.csv.
- flat/ 및 mass8/: collections/*/rollout.h5, summary.json, training/*/checkpoint와 학습 이력, evaluations/*/seed별 원본.
- flat/aggregate/ 및 mass8/aggregate/: seed_results.csv (원본 절대 경로 포함), condition_summary.csv, paired_seed_differences.csv, paired_summary.csv.
- mass8/aggregate/: chain_comparison.csv, collection_seed_stats.csv, collection_episode_stats.csv.
- flat/losses.csv 및 mass8/losses.csv: 모델·seed별 loss와 원본 H5 경로. *_summary.csv: seed 평균/SD와 pooled window 평균.
- commands.jsonl, logs/: 실행 명령과 로그; source_unchanged_audit.csv: 기존 파일 hash 불변 감사.
