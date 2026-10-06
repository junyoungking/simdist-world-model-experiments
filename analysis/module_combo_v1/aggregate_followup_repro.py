"""Aggregate second-chain reproduction and +8kg D_P evaluations."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import csv
import json
import statistics as st
from pathlib import Path
ROOT = artifact_dir('module_combo_v1')
REPO = PORTFOLIO_ROOT
RESULTS = ROOT
FOLLOW = RESULTS / 'followup_repro'
SEEDS = range(23501, 23601)
FIELDS = ('fall', 'episode_length', 'return', 'reward_per_step')

def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def raw_row(path, gap, model, seed):
    raw = json.loads(resolve_artifact_path(path).read_text())
    assert raw['status'] == 'evaluated' and raw['seed'] == seed
    assert raw['forward_target_mps'] == 1.0 and raw['warmup_termination_count'] == 0
    assert raw['episode_length'] == 1000 or raw['fall'] == 1
    return {'gap': gap, 'model': model, 'seed': seed, **{field: raw[field] for field in FIELDS}, 'end_reason': raw['end_reason'], 'raw_summary': str(resolve_artifact_path(path).resolve())}

def paired(left_name, right_name, left, right, gap):
    selected = []
    for seed in SEEDS:
        a, b = (left[seed], right[seed])
        row = {'gap': gap, 'comparison': f'{left_name}-minus-{right_name}', 'seed': seed, 'left_fail': a['fall'], 'right_fail': b['fall']}
        row.update({field + '_difference': a[field] - b[field] for field in FIELDS})
        selected.append(row)
    summary = {'gap': gap, 'comparison': f'{left_name}-minus-{right_name}', 'seeds': len(selected), 'both_fail': sum((r['left_fail'] and r['right_fail'] for r in selected)), 'left_only_fail': sum((r['left_fail'] and (not r['right_fail']) for r in selected)), 'right_only_fail': sum((not r['left_fail'] and r['right_fail'] for r in selected)), 'both_survive': sum((not r['left_fail'] and (not r['right_fail']) for r in selected))}
    for field in FIELDS:
        diffs = [row[field + '_difference'] for row in selected]
        summary[field + '_mean_difference'] = st.mean(diffs)
        summary[field + '_sd_difference'] = st.stdev(diffs)
    return (selected, summary)

def condition_summary(gap, data):
    result = []
    for model, by_seed in data.items():
        chosen = [by_seed[seed] for seed in SEEDS]
        row = {'gap': gap, 'model': model, 'seeds': len(chosen), 'failures': sum((item['fall'] for item in chosen))}
        for field in FIELDS[1:]:
            values = [item[field] for item in chosen]
            row[field + '_mean'] = st.mean(values)
            row[field + '_sd'] = st.stdev(values)
        result.append(row)
    return result

def load_new_eval(root, gap, model, controller, seed):
    name = f'{controller}_{gap}_v10_seed{seed}'
    return root / name / 'summary.json'

def eval_repro_gap(gap):
    out = FOLLOW / gap / 'second_chain'
    source_old = {int(row['seed']): row for row in read_csv(RESULTS / gap / 'aggregate/seed_results.csv') if row['model'] == 'n0p0_mppi'}
    chain1_old = {int(row['seed']): row for row in read_csv(RESULTS / gap / 'aggregate/seed_results.csv') if row['model'] == 'd30p0_mppi'}
    by_model = {'m0': {}, 'chain1_30k': {}, 'chain2_30k': {}}
    for seed in SEEDS:
        for model, legacy in (('m0', source_old), ('chain1_30k', chain1_old)):
            if seed < 23551:
                prior = legacy[seed]
                by_model[model][seed] = {'gap': gap, 'model': model, 'seed': seed, **{f: int(prior[f]) if f == 'fall' else float(prior[f]) for f in FIELDS}, 'end_reason': prior['end_reason'], 'raw_summary': prior['raw_summary']}
            else:
                cond = 'm0_extension' if model == 'm0' else 'chain1_30k_extension'
                path = load_new_eval(out / 'evaluations' / cond, gap, model, 'mppi', seed)
                by_model[model][seed] = raw_row(path, gap, model, seed)
        path = load_new_eval(out / 'evaluations/chain2_30k', gap, 'chain2_30k', 'mppi', seed)
        by_model['chain2_30k'][seed] = raw_row(path, gap, 'chain2_30k', seed)
    agg = out / 'aggregate'
    all_rows = [row for items in by_model.values() for row in items.values()]
    write_csv(agg / 'seed_results.csv', all_rows)
    write_csv(agg / 'condition_summary.csv', condition_summary(gap, by_model))
    pair_rows, pair_summaries = ([], [])
    for left, right in (('chain1_30k', 'm0'), ('chain2_30k', 'm0'), ('chain2_30k', 'chain1_30k')):
        rows, summary = paired(left, right, by_model[left], by_model[right], gap)
        pair_rows.extend(rows)
        pair_summaries.append(summary)
    write_csv(agg / 'paired_seed_differences.csv', pair_rows)
    write_csv(agg / 'paired_summary.csv', pair_summaries)
    return agg

def eval_mass8():
    root = FOLLOW / 'mass8_d_p'
    old_rows = read_csv(RESULTS / 'mass8/aggregate/seed_results.csv')
    old = {model: {int(row['seed']): row for row in old_rows if row['model'] == model} for model in ('d30p50_mppi', 'n0p50_mppi', 'd30p0_mppi')}
    models = {key: {} for key in ('D_P__P', 'D_P__Pnone', 'D_P__P_baseline', 'D__P', 'D__Pnone')}
    source_map = {'D_P__P_baseline': 'd30p50_mppi', 'D__P': 'n0p50_mppi', 'D__Pnone': 'd30p0_mppi'}
    for seed in SEEDS:
        for label, legacy_model in source_map.items():
            prior = old[legacy_model][seed]
            models[label][seed] = {'gap': 'mass8', 'model': label, 'seed': seed, **{f: int(prior[f]) if f == 'fall' else float(prior[f]) for f in FIELDS}, 'end_reason': prior['end_reason'], 'raw_summary': prior['raw_summary']}
        for label, dirname in (('D_P__P', 'd_p_mppi'), ('D_P__Pnone', 'd_pnone_mppi')):
            path = load_new_eval(root / 'evaluations' / dirname, 'mass8', label, 'mppi', seed)
            models[label][seed] = raw_row(path, 'mass8', label, seed)
    agg = root / 'aggregate'
    write_csv(agg / 'seed_results.csv', [row for group in models.values() for row in group.values()])
    write_csv(agg / 'condition_summary.csv', condition_summary('mass8', models))
    pair_rows, pair_summaries = ([], [])
    comparisons = (('D_P__P', 'D__P'), ('D_P__P', 'D_P__P_baseline'), ('D_P__P', 'D__Pnone'), ('D_P__Pnone', 'D__P'), ('D_P__Pnone', 'D_P__P_baseline'), ('D_P__Pnone', 'D__Pnone'), ('D_P__P', 'D_P__Pnone'))
    for left, right in comparisons:
        rows, summary = paired(left, right, models[left], models[right], 'mass8')
        pair_rows.extend(rows)
        pair_summaries.append(summary)
    write_csv(agg / 'paired_seed_differences.csv', pair_rows)
    write_csv(agg / 'paired_summary.csv', pair_summaries)
    return agg

def main():
    agg_paths = [eval_repro_gap('friction015'), eval_repro_gap('gain070'), eval_mass8()]
    report = ['# Dynamics 재현 및 +8kg D_P 진단', '', '평가 목표 1.0 m/s, 규약 v2. 길이·return·reward/step은 seed 평균 ± 표준편차.', '']
    for gap in ('friction015', 'gain070'):
        agg = FOLLOW / gap / 'second_chain/aggregate'
        summaries = read_csv(agg / 'condition_summary.csv')
        report.extend([f'## {gap}: 두 번째 dynamics 체인', '', '| 조건 | 실패 | 길이 | return | reward/step |', '|---|---:|---:|---:|---:|'])
        for row in summaries:
            report.append(f"| {row['model']} | {row['failures']}/100 | {float(row['episode_length_mean']):.1f} ± {float(row['episode_length_sd']):.1f} | {float(row['return_mean']):.3f} ± {float(row['return_sd']):.3f} | {float(row['reward_per_step_mean']):.4f} ± {float(row['reward_per_step_sd']):.4f} |")
        pair_summaries = read_csv(agg / 'paired_summary.csv')
        report.extend(['', '| 짝 비교 | Δ 실패율 | Δ 길이 | Δ return | Δ reward/step | 실패 교차: 둘/왼쪽만/오른쪽만/생존 |', '|---|---:|---:|---:|---:|---:|'])
        for row in pair_summaries:
            report.append(f"| {row['comparison']} | {float(row['fall_mean_difference']):.3f} ± {float(row['fall_sd_difference']):.3f} | {float(row['episode_length_mean_difference']):.1f} ± {float(row['episode_length_sd_difference']):.1f} | {float(row['return_mean_difference']):.3f} ± {float(row['return_sd_difference']):.3f} | {float(row['reward_per_step_mean_difference']):.4f} ± {float(row['reward_per_step_sd_difference']):.4f} | {row['both_fail']}/{row['left_only_fail']}/{row['right_only_fail']}/{row['both_survive']} |")
        loss = read_csv(FOLLOW / gap / 'second_chain/losses_summary.csv') if (FOLLOW / gap / 'second_chain/losses_summary.csv').exists() else None
        if loss:
            report.extend(['', '| 모델 | 1-step MSE | 25-step MSE |', '|---|---:|---:|'])
            by_model = {r['model']: r for r in loss if r['metric'] == 'full_25_step_latent_mse'}
            one = {r['model']: r for r in loss if r['metric'] == 'one_step_latent_mse'}
            for model in ('m0', 'chain1_30k', 'chain2_30k'):
                a, b = (one[model], by_model[model])
                report.append(f"| {model} | {float(a['seed_mean']):.5f} ± {float(a['seed_sample_sd']):.5f} | {float(b['seed_mean']):.5f} ± {float(b['seed_sample_sd']):.5f} |")
        report.extend(['', f"원본 경로: `{FOLLOW / gap / 'second_chain'}`.", ''])
    agg = FOLLOW / 'mass8_d_p/aggregate'
    summaries = read_csv(agg / 'condition_summary.csv')
    report.extend(['## +8kg: D_P 진단', '', '| 조건 | 실패 | 길이 | return | reward/step |', '|---|---:|---:|---:|---:|'])
    for row in summaries:
        report.append(f"| {row['model']} | {row['failures']}/100 | {float(row['episode_length_mean']):.1f} ± {float(row['episode_length_sd']):.1f} | {float(row['return_mean']):.3f} ± {float(row['return_sd']):.3f} | {float(row['reward_per_step_mean']):.4f} ± {float(row['reward_per_step_sd']):.4f} |")
    pair_summaries = read_csv(agg / 'paired_summary.csv')
    report.extend(['', '| 짝 비교 | Δ 실패율 | Δ 길이 | Δ return | Δ reward/step | 실패 교차: 둘/왼쪽만/오른쪽만/생존 |', '|---|---:|---:|---:|---:|---:|'])
    for row in pair_summaries:
        report.append(f"| {row['comparison']} | {float(row['fall_mean_difference']):.3f} ± {float(row['fall_sd_difference']):.3f} | {float(row['episode_length_mean_difference']):.1f} ± {float(row['episode_length_sd_difference']):.1f} | {float(row['return_mean_difference']):.3f} ± {float(row['return_sd_difference']):.3f} | {float(row['reward_per_step_mean_difference']):.4f} ± {float(row['reward_per_step_sd_difference']):.4f} | {row['both_fail']}/{row['left_only_fail']}/{row['right_only_fail']}/{row['both_survive']} |")
    loss = read_csv(FOLLOW / 'mass8_d_p/losses_summary.csv')
    report.extend(['', '| 데이터 | 모델 | 25-step MSE |', '|---|---|---:|'])
    for row in loss:
        if row['metric'] == 'full_25_step_latent_mse':
            report.append(f"| {row['dataset']} | {row['model']} | {float(row['seed_mean']):.5f} ± {float(row['seed_sample_sd']):.5f} |")
    report.extend(['', 'D_P는 m0+P checkpoint에서 시작해 policy/policy_dec을 고정하고 dynamics만 두 단계 학습했다. D_P,P 없음 평가는 D_P dynamics에 원래 m0 policy/policy_dec을 이식한 checkpoint로 수행했다.', '', f"원본 경로: `{FOLLOW / 'mass8_d_p'}`.", '', '## 산출물', '', f'평가 seed 요약과 원본 summary.json 경로: `{FOLLOW}` 아래 gap별 `aggregate/seed_results.csv`.', '조건 평균, 짝 seed 차이, 실패 교차표는 gap별 `aggregate/` CSV에 있다.', ''])
    (ROOT / 'FOLLOWUP_REPORT.md').write_text('\n'.join(report))
    print(ROOT / 'FOLLOWUP_REPORT.md')
if __name__ == '__main__':
    main()
