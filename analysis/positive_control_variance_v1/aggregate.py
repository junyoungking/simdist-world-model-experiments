"""Complete tables, paired seeds, collection windows and unchanged-source audit."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import csv
import h5py
import hashlib
import itertools
import json
import math
import statistics as st
import time
from pathlib import Path
ROOT = artifact_dir('positive_control_variance_v1')
REPO = PORTFOLIO_ROOT
R = ROOT
SEEDS = range(23501, 23601)
FIELDS = ('failure', 'episode_length', 'return', 'reward_per_step', 'mean_vx_body_mps', 'mean_vx_abs_tracking_error_mps')

def read(path):
    with resolve_artifact_path(path).open() as f:
        return list(csv.DictReader(f))

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def raw(path, model):
    s = json.loads(resolve_artifact_path(path).read_text())
    assert s['status'] in ('evaluated', 'warmup_failed')
    assert s['joint_reset_max_error'] < 1e-05 and s['joint_velocity_reset_max_error'] < 1e-05
    assert s['episode_length'] == 1000 or s['fall'] or s['status'] == 'warmup_failed'
    return {'model': model, 'seed': s['seed'], 'failure': int(s['fall'] or s['status'] == 'warmup_failed'), **{k: s[k] for k in FIELDS[1:]}, 'warmup_failed': int(s['status'] == 'warmup_failed'), 'end_reason': s['end_reason'], 'raw_summary': str(resolve_artifact_path(path).resolve())}

def mean_sd(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return (st.mean(values), st.stdev(values) if len(values) > 1 else 0.0) if values else (None, None)

def evaluations(out, paths):
    data = {model: {raw(p, model)['seed']: raw(p, model) for p in files} for model, files in paths.items()}
    assert all((set(by_seed) == set(SEEDS) for by_seed in data.values()))
    write(out / 'seed_results.csv', [v for by_seed in data.values() for v in by_seed.values()])
    summaries = []
    for model, by_seed in data.items():
        row = {'model': model, 'seeds': 100, 'failures': sum((r['failure'] for r in by_seed.values())), 'warmup_failures': sum((r['warmup_failed'] for r in by_seed.values()))}
        for f in FIELDS[1:]:
            row[f + '_mean'], row[f + '_sd'] = mean_sd((r[f] for r in by_seed.values()))
        summaries.append(row)
    write(out / 'condition_summary.csv', summaries)
    differences, cross = ([], [])
    pairs = [(m, 'm0') for m in data if m != 'm0']
    pairs += list(itertools.combinations([m for m in data if m != 'm0'], 2))
    for left, right in pairs:
        paired = []
        for seed in SEEDS:
            a, b = (data[left][seed], data[right][seed])
            row = {'comparison': left + '-minus-' + right, 'seed': seed, 'left_failure': a['failure'], 'right_failure': b['failure'], 'left_raw': a['raw_summary'], 'right_raw': b['raw_summary']}
            for f in FIELDS:
                row[f + '_difference'] = a[f] - b[f] if a[f] is not None and b[f] is not None else None
            paired.append(row)
        summary = {'comparison': left + '-minus-' + right, 'seeds': 100, 'both_fail': sum((r['left_failure'] and r['right_failure'] for r in paired)), 'left_only_fail': sum((r['left_failure'] and (not r['right_failure']) for r in paired)), 'right_only_fail': sum((not r['left_failure'] and r['right_failure'] for r in paired)), 'both_survive': sum((not r['left_failure'] and (not r['right_failure']) for r in paired))}
        for f in FIELDS:
            summary[f + '_mean_difference'], summary[f + '_sd_difference'] = mean_sd((r[f + '_difference'] for r in paired))
        cross.append(summary)
        differences.extend(paired)
    write(out / 'paired_seed_differences.csv', differences)
    write(out / 'paired_summary.csv', cross)
    return (summaries, cross)

def historical(model, csv_path, legacy):
    rows = [r for r in read(csv_path) if r['model'] == legacy]
    assert len(rows) == 100
    return [resolve_artifact_path(r['raw_summary']) for r in rows]

def collection_stats(label, policy, roots):
    rows, episodes = ([], [])
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob('summary.json'))
        for file in files:
            s = json.loads(file.read_text())
            with h5py.File(file.with_name('rollout.h5'), 'r') as store:
                groups = [store[k] for k in sorted(store)]
                assert len(groups) == len(s['episodes'])
                for group, episode in zip(groups, s['episodes']):
                    assert group['actions'].shape == (episode['length'], 12)
                    assert group['proprio_obs'].shape == (episode['length'], 35)
                    assert group['extero_obs'].shape == (episode['length'], 315)
                    assert group.attrs['end_reason'] == episode['end_reason']
            assert s['completed_control_steps'] == sum((e['length'] for e in s['episodes']))
            assert s['falls'] == sum((e['end_reason'] == 'terminated' for e in s['episodes']))
            n_windows = near = overlaps = near_steps = 0
            for ep in s['episodes']:
                length = ep['length']
                windows = max(0, length - 50)
                falls = ep['end_reason'] == 'terminated'
                anchors = range(25, length - 25)
                proximity = sum((1 <= length - t <= 100 for t in anchors)) if falls else 0
                overlap = sum((length - (t + 25) <= 100 for t in anchors)) if falls else 0
                n_windows += windows
                near += proximity
                overlaps += overlap
                near_steps += min(100, length) if falls else 0
                episodes.append({'chain': label, 'seed': s['seed'], 'episode': ep['index'], 'length': length, 'end_reason': ep['end_reason'], 'valid_windows': windows, 'near_fall_anchor_windows': proximity, 'near_fall_overlap_windows': overlap, 'raw_summary': str(file.resolve())})
            rows.append({'chain': label, 'policy': policy, 'seed': s['seed'], 'control_steps': s['completed_control_steps'], 'falls': s['falls'], 'warmup_terminations': s['warmup_terminations'], 'valid_windows': n_windows, 'near_fall_anchor_windows': near, 'near_fall_overlap_windows': overlaps, 'near_fall_control_steps': near_steps, 'checkpoint': s['checkpoint'], 'raw_summary': str(file.resolve())})
    summary = {'chain': label, 'collection_policy': policy}
    for key in ('control_steps', 'falls', 'warmup_terminations', 'valid_windows', 'near_fall_anchor_windows', 'near_fall_overlap_windows', 'near_fall_control_steps'):
        summary[key] = sum((r[key] for r in rows))
    assert summary['control_steps'] == 30000
    summary['near_fall_window_ratio'] = summary['near_fall_anchor_windows'] / summary['valid_windows']
    summary['near_fall_overlap_ratio'] = summary['near_fall_overlap_windows'] / summary['valid_windows']
    summary['near_fall_step_ratio'] = summary['near_fall_control_steps'] / summary['control_steps']
    return (summary, rows, episodes)

def fmt(value, sd, digits=3):
    return f'{value:.{digits}f} ± {sd:.{digits}f}' if value is not None else 'NA'

def paired_table(rows):
    lines = ['| 비교 (왼쪽−오른쪽) | Δ 실패율 | Δ 길이 | Δ return | Δ reward/step | 실패 교차: 둘/왼쪽만/오른쪽만/생존 |', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        values = [fmt(r[f + '_mean_difference'], r[f + '_sd_difference'], d) for f, d in (('failure', 3), ('episode_length', 1), ('return', 3), ('reward_per_step', 4))]
        cross = '/'.join((str(r[f]) for f in ('both_fail', 'left_only_fail', 'right_only_fail', 'both_survive')))
        lines.append('| ' + ' | '.join([r['comparison'], *values, cross]) + ' |')
    return lines

def main():
    selection = json.loads((R / 'selection.json').read_text())
    screen_rows = []
    for path in sorted((R / 'screen').rglob('summary.json')):
        s = json.loads(path.read_text())
        screen_rows.append({'condition': path.parents[2].name, 'controller': s['controller'], 'seed': s['seed'], 'friction': s['foot_friction'], 'target_mps': s['forward_target_mps'], 'failure': int(s['fall'] or s['status'] == 'warmup_failed'), 'episode_length': s['episode_length'], 'return': s['return'], 'raw_summary': str(path.resolve())})
    assert len(screen_rows) == 240
    write(R / 'screen_seed_results.csv', screen_rows)
    flat, mass = (R / 'flat', R / 'mass8')
    flat_summary, flat_pairs = evaluations(flat / 'aggregate', {name: sorted((flat / 'evaluations' / name).rglob('summary.json')) for name in ('m0', 'chain1', 'chain2', 'base')})
    paths = {'m0': historical('m0', result_path('module_combo_v1/mass8/aggregate/seed_results.csv'), 'n0p0_mppi'), 'old_chain1': historical('old_chain1', result_path('module_combo_v1/mass8/aggregate/seed_results.csv'), 'd30p0_mppi'), 'old_chain2': historical('old_chain2', result_path('followup_v1/chain/aggregate/seed_results.csv'), 'new_30k'), 'old_DP': historical('old_DP', result_path('module_combo_v1/followup_repro/mass8_d_p/aggregate/seed_results.csv'), 'D_P__Pnone')}
    for name in ('new_chain3', 'new_chain4', 'new_DP2'):
        paths[name] = sorted((mass / 'evaluations' / name).rglob('summary.json'))
    mass_summary, mass_pairs = evaluations(mass / 'aggregate', paths)
    old = artifact_root()
    sources = {'old_chain1': [old / 'adaptation_curve_v1/collections' / stage / 'summary.json' for stage in ('stage_1000', 'stage_3000', 'stage_10000', 'stage_30000')], 'old_chain2': [old / 'followup_v1/chain/collections'], 'old_DP': [old / 'module_combo_v1/followup_repro/mass8_d_p/collections'], **{label: [mass / label / 'collections'] for label in ('new_chain3', 'new_chain4', 'new_DP2')}}
    collections, raw_collections, episode_rows = ([], [], [])
    for label, roots in sources.items():
        policy = 'MPPI (D_stage, P50)' if label in ('old_DP', 'new_DP2') else 'MPPI (D_stage, original P0)'
        summary, rows, episodes = collection_stats(label, policy, roots)
        collections.append(summary)
        raw_collections.extend(rows)
        episode_rows.extend(episodes)
    write(mass / 'aggregate/collection_seed_stats.csv', raw_collections)
    write(mass / 'aggregate/collection_episode_stats.csv', episode_rows)
    loss = {r['model']: r for r in read(mass / 'losses_summary.csv') if r['metric'] == 'full_25_step_latent_mse'}
    failure = {r['model']: r['failures'] for r in mass_summary}
    for r in collections:
        r['independent_25step_mean'] = float(loss[r['chain']]['seed_mean'])
        r['independent_25step_sd'] = float(loss[r['chain']]['seed_sample_sd'])
        r['failures_100'] = failure[r['chain']]
    write(mass / 'aggregate/chain_comparison.csv', collections)
    training = []
    for path in sorted(R.rglob('training/*/summary.json')):
        s = json.loads(path.read_text())
        assert not s['frozen_parameter_changes'] and s['dynamics_parameter_change_count'] > 0
        training.append({'chain': str(path.parent.parent.parent.relative_to(R)), 'stage': path.parent.name, 'updates': s['actual_updates'], 'selected_update': s['selected_best_update'], 'train_windows': s['cumulative_train_windows'], 'heldout_windows': s['cumulative_heldout_windows'], 'heldout_loss': s['heldout_all_loss'], 'raw_summary': str(path)})
    assert len(training) == 10
    write(R / 'training_summary.csv', training)
    before = json.loads((ROOT / 'source_hashes_before.json').read_text())
    audit = [{'path': p, 'sha256_before': h, 'sha256_after': hashlib.sha256(resolve_artifact_path(p).read_bytes()).hexdigest() if resolve_artifact_path(p).is_file() else 'MISSING'} for p, h in before.items()]
    for r in audit:
        r['unchanged'] = r['sha256_before'] == r['sha256_after']
    write(ROOT / 'source_unchanged_audit.csv', audit)
    assert all((r['unchanged'] for r in audit))
    runtime = {'initial_expected_hours': '5-6', 'initial_conservative_hours': 7, 'updated_expected_hours_after_plane_screen': '7-8', 'elapsed_hours_at_report': (time.time() - (ROOT / 'source_hashes_before.json').stat().st_mtime) / 3600, 'part2_executed': True}
    (ROOT / 'runtime_summary.json').write_text(json.dumps(runtime, indent=2) + '\n')
    reason = '선정 기준을 충족했다.' if selection['eligible'] else '적격 조건이 없어 expert 실패≤1 및 m0 실패 8–17까지의 실패 수 거리 합이 최소인 조건을 선택했다.'
    lines = ['# 평지 양성 대조군과 +8kg dynamics 체인 편차', '', f"초기 예상 5–6시간, 평지 실측 반영 후 7–8시간. 전체 예상이 9시간 이하여서 2번도 실행했다. 실제 소요 약 {runtime['elapsed_hours_at_report']:.2f}시간.", '', '모든 새 코드와 결과는 experiments/positive_control_variance_v1에 작성했다. 기존 파일의 SHA256 불변 확인을 통과했다.', '', '## 규약 및 지형', '', 'v2: 기본 관절 위치·속도 reset, PPO expert 25-step warm-up, 목표 주행 최대 1,000 step (50 Hz). 실패는 평가 terminated 또는 warm-up 실패다. 표의 ±는 seed 간 표본 SD다. 목표 속도는 기존 경로 추종 command의 forward_vel이며 제어 규약은 그대로다.', '', "평지는 IsaacLab TerrainImporterCfg의 terrain_type='plane', terrain_generator=None인 무한 평면이다. height scanner와 433-dim expert 관측, 35/315-dim 모델 관측 및 reward는 유지했다. 초기 ray hit world-z span <1e-5 m를 매 seed 확인했다. 질량 +0kg, Kp/Kd=25/0.5, 발 restitution=0.05. +8kg은 기존 boxes difficulty=0.4, 마찰 0.8, 목표 1.0m/s다.", '', '## 1. 학습 전 평지 조건 탐색 (23501–23520)', '', '| 발 마찰 | 목표 m/s | m0 MPPI 실패 | expert 실패 | 적격 | 범위까지 거리 |', '|---:|---:|---:|---:|---|---:|']
    for r in selection['candidates']:
        lines.append(f"| {r['friction']:.2f} | {r['velocity']:.1f} | {r['mppi_failures']}/20 | {r['expert_failures']}/20 | {r['eligible']} | {r['distance']} |")
    lines.extend(['', f"선택: 마찰 {selection['friction']:.2f}, 목표 {selection['velocity']:.1f}m/s. 이유: {reason}", '거리 = max(expert 실패−1,0) + max(8−m0 실패,m0 실패−17,0). 동률이면 마찰 0.15/0.5m/s 우선, 이후 expert 실패 수, m0 실패 수의 구간 중앙(12.5)까지 거리, 조건명으로 정했다.', '', '## 평지 100-seed 평가 (23501–23600)', '', '| 모델 | 실패 | 길이 | return | reward/step |', '|---|---:|---:|---:|---:|'])
    for r in flat_summary:
        lines.append(f"| {r['model']} | {r['failures']}/100 | {fmt(r['episode_length_mean'], r['episode_length_sd'], 1)} | {fmt(r['return_mean'], r['return_sd'])} | {fmt(r['reward_per_step_mean'], r['reward_per_step_sd'], 4)} |")
    lines.extend(['', *paired_table(flat_pairs), '', '独립 검증: m0 MPPI 10 seed (45301–45310) × 500 control step. 학습·중단 기준에 사용하지 않았다.', '', '| 모델 | 1-step latent MSE | 25-step latent MSE |', '|---|---:|---:|'])
    flat_loss = read(flat / 'losses_summary.csv')
    for label in ('m0', 'chain1', 'chain2'):
        rows = {r['metric']: r for r in flat_loss if r['model'] == label}
        values = [fmt(float(rows[k]['seed_mean']), float(rows[k]['seed_sample_sd']), 5) for k in ('one_step_latent_mse', 'full_25_step_latent_mse')]
        lines.append('| ' + ' | '.join([label, *values]) + ' |')
    lines.extend(['', '평지 base policy는 100개 seed에서 모두 길이 532 step이었다. 평면·고정 물성과 deterministic head에서 같은 결과가 반복됐으므로 이 seed 반복을 서로 다른 지형 표본으로 해석하지 않는다.', '', '평지 두 체인은 43101–43106/43201–43206, 44101–44106/44201–44206 수집; 학습 seed 43501/43502, 44501/44502. 단계별 6개 수집 seed 중 마지막은 held-out이며 train/held-out을 단계 간 누적했다.', '', '## 2. +8kg 체인별 수집·loss·실패', '', '평가 시 모든 dynamics에 원래 policy P0를 사용했다. D_P 계열은 P50를 고정하여 수집·학습하고 평가용 checkpoint에는 P0를 이식했다. 수집 총량은 체인당 30,000 control step. warm-up은 수집 step과 window에서 제외했다.', '', '| 체인 | 수집 정책 | 넘어짐 | warm-up 넘어짐 | 근접 window / 전체 (%) | 독립 25-step MSE | 실패 |', '|---|---|---:|---:|---:|---:|---:|'])
    for r in collections:
        lines.append(f"| {r['chain']} | {r['collection_policy']} | {r['falls']} | {r['warmup_terminations']} | {r['near_fall_anchor_windows']}/{r['valid_windows']} ({100 * r['near_fall_window_ratio']:.2f}%) | {fmt(r['independent_25step_mean'], r['independent_25step_sd'], 5)} | {r['failures_100']}/100 |")
    lines.extend(['', f"m0: 실패 {failure['m0']}/100, 독립 25-step MSE {fmt(float(loss['m0']['seed_mean']), float(loss['m0']['seed_sample_sd']), 5)}.", '', '근접 window: L개 control step인 episode에서 유효 window start=0..L−51, 예측 anchor t=start+25. terminated episode에 대해 1≤L−t≤100인 window 수 / 모든 유효 window 수. 동일 window 규칙으로 체인 간 계산했다. overlap 기준과 control-step 기준 비율도 collection CSV에 보조로 저장했다. timeout과 budget-censored의 끝을 넘어짐으로 세지 않았다.', '', '독립 세트는 새 m0 MPPI seed 49301–49310 × 500 step (mass8/validation)이며, 모든 모델을 동일 episode/window에 재평가했다. 기존 m0 검증 세트의 seed 35001은 첫 체인의 후속 30k_c 분기에서 cp10000 중단용 자료와 simulator seed가 같아 사용하지 않고 새 세트를 수집했다.', '', 'old_chain1은 기존 첫 30k checkpoint (stage_30000, step 380304)와 기존 4개 1k/2k/7k/20k 수집 파일; old_chain2는 followup_v1의 두 번째 체인; old_DP는 module_combo_v1/followup_repro/mass8_d_p다. 첫 체인은 기존 수집 단계·split이 새 체인과 다르므로 완전히 대칭적인 반복 실험은 아니다.', '', '새 P0 체인 수집 seed: 46101–46106/46201–46206 및 47101–47106/47201–47206; 학습 46501/46502 및 47501/47502. 새 D_P: 48101–48106/48201–48206; 학습 48501/48502.', '', '## +8kg m0 대비 같은 seed 차이 및 실패 교차표', '', *paired_table([r for r in mass_pairs if r['comparison'].endswith('-minus-m0')]), '', '전체 체인 간 짝 비교는 mass8/aggregate/paired_summary.csv 및 paired_seed_differences.csv에 있다.', '', '## 학습 규칙 및 원본', '', 'dynamics-only latent loss, batch 32, 기존 warm-up cosine Adam, 단계마다 fresh optimizer. 10 epoch update cap; 100-update 간격 held-out 검사; 최소 500 update; 상대 0.1% 개선 없는 검사 5회 시 중단; best checkpoint 선택. 모든 새 학습에서 frozen parameter 변경 0 확인.', '', '| 체인/단계 | train/held-out window | 선택/실행 update | held-out loss |', '|---|---:|---:|---:|'])
    for r in training:
        lines.append(f"| {r['chain']}/{r['stage']} | {r['train_windows']}/{r['heldout_windows']} | {r['selected_update']}/{r['updates']} | {r['heldout_loss']:.5f} |")
    lines.extend(['', f'원본 root: `{R}`.', '', '- screen_seed_results.csv 및 screen/: 6조건 × 2제어기 × 20 seed summary.json, warmup_steps.csv, evaluation_steps.csv.', '- flat/ 및 mass8/: collections/*/rollout.h5, summary.json, training/*/checkpoint와 학습 이력, evaluations/*/seed별 원본.', '- flat/aggregate/ 및 mass8/aggregate/: seed_results.csv (원본 절대 경로 포함), condition_summary.csv, paired_seed_differences.csv, paired_summary.csv.', '- mass8/aggregate/: chain_comparison.csv, collection_seed_stats.csv, collection_episode_stats.csv.', '- flat/losses.csv 및 mass8/losses.csv: 모델·seed별 loss와 원본 H5 경로. *_summary.csv: seed 평균/SD와 pooled window 평균.', '- commands.jsonl, logs/: 실행 명령과 로그; source_unchanged_audit.csv: 기존 파일 hash 불변 감사.', ''])
    (ROOT / 'REPORT.md').write_text('\n'.join(lines).replace('独립', '독립'))
    print(ROOT / 'REPORT.md')
if __name__ == '__main__':
    main()
