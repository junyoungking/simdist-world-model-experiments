"""Aggregate matched-seed module combinations and produce a compact report."""
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
GAPS = {'mass8': 100, 'friction015': 50, 'gain070': 50}
MODELS = ('n0p0_mppi', 'd30p0_mppi', 'n0p50_mppi', 'd30p50_mppi', 'n0p0_base', 'n0p50_base')
FIELDS = ('fall', 'episode_length', 'return', 'reward_per_step')
PAIRS = (('d30p0_mppi', 'n0p0_mppi'), ('n0p50_mppi', 'n0p0_mppi'), ('d30p50_mppi', 'd30p0_mppi'), ('d30p50_mppi', 'n0p50_mppi'))

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def stats(values):
    return (st.mean(values), st.stdev(values))

def previous_index(path, model):
    with path.open() as stream:
        return {int(row['seed']): resolve_artifact_path(row['raw_summary']) for row in csv.DictReader(stream) if row['model'] == model}

def raw_path(gap, model, seed):
    if gap == 'mass8' and model in ('n0p0_mppi', 'd30p0_mppi', 'd90p0_mppi'):
        name = {'n0p0_mppi': 'm0', 'd30p0_mppi': 'cp30000', 'd90p0_mppi': '90k'}[model]
        index = previous_index(result_path('extended_adaptation_v1/aggregate/seed_results.csv'), name)
        return index[seed]
    if gap == 'mass8' and model == 'n0p0_base':
        index = previous_index(artifact_root() / 'extended_adaptation_v1/oracle/aggregate/seed_results.csv', 'old_base')
        return index[seed]
    if gap == 'mass8':
        evaluation_name = {'n0p50_mppi': 'm0p_mppi', 'd30p50_mppi': 'd30p_mppi', 'n0p50_base': 'm0p_base', 'd90p50_mppi': 'd90p_mppi'}.get(model, model)
        controller = 'base' if model.endswith('_base') else 'mppi'
        return RESULTS / gap / 'evaluations' / evaluation_name / f'{controller}_mass8_v10_seed{seed}/summary.json'
    controller = 'base' if model.endswith('_base') else 'mppi'
    return RESULTS / gap / 'evaluations' / model / f'{controller}_{gap}_v10_seed{seed}/summary.json'

def load_rows(gap, n):
    rows, by_key = ([], {})
    models = (*MODELS, 'd90p0_mppi', 'd90p50_mppi') if gap == 'mass8' else MODELS
    for model in models:
        for seed in range(23501, 23501 + n):
            path = raw_path(gap, model, seed)
            raw = json.loads(path.read_text())
            assert raw['status'] == 'evaluated' and raw['seed'] == seed
            assert raw['warmup_termination_count'] == 0
            assert raw['forward_target_mps'] == 1.0
            assert raw['episode_length'] == 1000 or raw['fall'] == 1
            row = {'gap': gap, 'model': model, 'seed': seed, **{key: raw[key] for key in FIELDS}, 'end_reason': raw['end_reason'], 'raw_summary': str(path.resolve())}
            rows.append(row)
            by_key[model, seed] = row
    return (rows, by_key, models)

def main():
    report = ['# 모듈 조합 비교', '', '목표 1.0 m/s, v2 reset, 같은 평가 seed를 짝지어 비교했다.', '']
    for gap, n in GAPS.items():
        rows, by_key, models = load_rows(gap, n)
        out = RESULTS / gap / 'aggregate'
        write_csv(out / 'seed_results.csv', rows)
        conditions = []
        for model in models:
            chosen = [by_key[model, seed] for seed in range(23501, 23501 + n)]
            summary = {'gap': gap, 'model': model, 'seeds': n, 'failures': sum((r['fall'] for r in chosen))}
            for field in FIELDS[1:]:
                mean, sd = stats([r[field] for r in chosen])
                summary[field + '_mean'] = mean
                summary[field + '_sd'] = sd
            conditions.append(summary)
        write_csv(out / 'condition_summary.csv', conditions)
        paired_rows, pair_summaries = ([], [])
        pairs = (*PAIRS, ('d90p50_mppi', 'd90p0_mppi')) if gap == 'mass8' else PAIRS
        for left, right in pairs:
            selected = []
            for seed in range(23501, 23501 + n):
                a, b = (by_key[left, seed], by_key[right, seed])
                row = {'gap': gap, 'comparison': f'{left}-minus-{right}', 'seed': seed, 'left_fail': a['fall'], 'right_fail': b['fall']}
                row.update({field + '_difference': a[field] - b[field] for field in FIELDS})
                selected.append(row)
                paired_rows.append(row)
            summary = {'gap': gap, 'comparison': f'{left}-minus-{right}', 'seeds': n, 'both_fail': sum((r['left_fail'] and r['right_fail'] for r in selected)), 'left_only_fail': sum((r['left_fail'] and (not r['right_fail']) for r in selected)), 'right_only_fail': sum((not r['left_fail'] and r['right_fail'] for r in selected)), 'both_survive': sum((not r['left_fail'] and (not r['right_fail']) for r in selected))}
            for field in FIELDS:
                mean, sd = stats([r[field + '_difference'] for r in selected])
                summary[field + '_mean_difference'] = mean
                summary[field + '_sd_difference'] = sd
            pair_summaries.append(summary)
        write_csv(out / 'paired_seed_differences.csv', paired_rows)
        write_csv(out / 'paired_summary.csv', pair_summaries)
        report.extend([f'## {gap} ({n} seed)', '', '| 조합 | 실패 | 길이 | return | reward/step |', '|---|---:|---:|---:|---:|'])
        for row in conditions:

            def pm(field, digits):
                return f"{row[field + '_mean']:.{digits}f} ± {row[field + '_sd']:.{digits}f}"
            report.append(f"| {row['model']} | {row['failures']}/{n} | {pm('episode_length', 1)} | {pm('return', 3)} | {pm('reward_per_step', 4)} |")
        report.extend(['', '| 짝 비교 (왼쪽−오른쪽) | Δ 실패율 | Δ 길이 | Δ return | Δ reward/step | 실패 교차: 둘/왼쪽만/오른쪽만/생존 |', '|---|---:|---:|---:|---:|---:|'])
        for row in pair_summaries:

            def pm_delta(field, digits):
                return f"{row[field + '_mean_difference']:.{digits}f} ± {row[field + '_sd_difference']:.{digits}f}"
            crossover = '/'.join((str(row[k]) for k in ('both_fail', 'left_only_fail', 'right_only_fail', 'both_survive')))
            report.append(f"| {row['comparison']} | {pm_delta('fall', 3)} | {pm_delta('episode_length', 1)} | {pm_delta('return', 3)} | {pm_delta('reward_per_step', 4)} | {crossover} |")
        if gap != 'mass8':
            loss = list(csv.DictReader((RESULTS / gap / 'losses/c_loss_summary.csv').open()))
            report.extend(['', '독립 m0 MPPI 수집: 10 seed × 500 step. 1-step / 25-step latent MSE (seed 평균 ± 표본 SD):', '', '| 모델 | 1-step | 25-step |', '|---|---:|---:|'])
            for row in loss:
                report.append(f"| {row['model']} | {float(row['one_step_latent_mse_seed_mean']):.5f} ± {float(row['one_step_latent_mse_seed_sample_sd']):.5f} | {float(row['full_25_step_latent_mse_seed_mean']):.5f} ± {float(row['full_25_step_latent_mse_seed_sample_sd']):.5f} |")
            report.extend(['', 'Dynamics 학습과 policy 학습:', '', '| 단계 | 수집 step | train/held-out window | 선택/실행 update | held-out loss |', '|---|---:|---:|---:|---:|'])
            for stage in ('10k', '30k'):
                summary = json.loads((RESULTS / gap / 'dynamics/training' / stage / 'summary.json').read_text())
                report.append(f"| D {stage} | {(10000 if stage == '10k' else 20000)} | {summary['cumulative_train_windows']}/{summary['cumulative_heldout_windows']} | {summary['selected_best_update']}/{summary['actual_updates']} | {summary['heldout_all_loss']:.5f} |")
            for stage in ('m0', '30k'):
                summary = json.loads((RESULTS / gap / 'policy' / stage / 'summary.json').read_text())
                saved = summary['saved'][0]
                report.append(f"| P from {stage} | 10000 expert | {summary['train_windows']}/{summary['heldout_windows']} | 50/50 | {saved['heldout_action_loss']:.5f} |")
        else:
            report.extend(['', 'D=기존 첫 체인 3만 checkpoint. 기존 m0·3만·base의 100-seed 평가를 재사용했다. P는 기존 +8kg expert 1만 step 데이터로 50 update 학습했다. 3만+P는 지난 oracle_50 checkpoint를 재사용해 100 seed로 평가했다.'])
        report.extend(['', f'원본: `results/{gap}/aggregate/seed_results.csv`, `condition_summary.csv`, `paired_seed_differences.csv`, `paired_summary.csv`; 수집·학습·평가는 `results/{gap}/` 아래에 있다.', ''])
    (ROOT / 'REPORT.md').write_text('\n'.join(report))
    print(ROOT / 'REPORT.md')
if __name__ == '__main__':
    main()
