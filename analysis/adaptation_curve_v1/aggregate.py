"""Combine v2 baseline episodes and adaptation-curve episodes by seed."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import csv
import json
import statistics as stats
from collections import Counter
from pathlib import Path
ROOT = artifact_dir('adaptation_curve_v1')
RESULTS = ROOT
OLD = artifact_dir('baseline_v2') / 'runs'
SEEDS = range(23501, 23521)
CONDITIONS = [('m0_target', 'mppi', 8, 'baseline_target'), ('base_target', 'base', 8, 'baseline_target'), ('expert_target', 'expert', 8, 'baseline_target'), ('m0_source', 'mppi', 0, 'baseline_source'), *[(f'cp{step}_target', 'mppi', 8, f'stage_{step}') for step in (1000, 3000, 10000, 30000)]]
METRICS = ('fall', 'episode_length', 'return', 'reward_per_step', 'mean_vx_body_mps', 'mean_vx_abs_tracking_error_mps')

def save_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    rows = []
    for label, controller, mass, run_dir in CONDITIONS:
        for seed in SEEDS:
            name = f'{controller}_mass{mass}_v10_seed{seed}'
            if run_dir.startswith('baseline') and seed <= 23510:
                path = OLD / name / 'summary.json'
            else:
                path = RESULTS / 'evaluations' / run_dir / name / 'summary.json'
            assert path.exists(), path
            record = json.loads(path.read_text())
            assert record['status'] == 'evaluated' and record['seed'] == seed
            assert record['controller'] == controller
            assert record['mass_added_kg'] == float(mass)
            assert record['forward_target_mps'] == 1.0
            assert record['warmup_steps'] == 25
            rows.append({'condition': label, 'seed': seed, **{metric: record[metric] for metric in METRICS}, 'end_reason': record['end_reason'], 'termination_terms': record['termination_terms'], 'warmup_termination_count': record['warmup_termination_count'], 'raw_summary': str(path.resolve())})
    out = RESULTS / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / 'seed_results.csv', rows)
    summary = []
    for label, _, _, _ in CONDITIONS:
        selected = [row for row in rows if row['condition'] == label]
        reasons = Counter(((row['end_reason'], row['termination_terms']) for row in selected))
        item = {'condition': label, 'n': len(selected), 'termination_reasons': json.dumps({'|'.join(k): v for k, v in reasons.items()})}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            item[metric + '_mean'] = stats.mean(values)
            item[metric + '_sample_sd'] = stats.stdev(values)
        summary.append(item)
    save_csv(out / 'condition_summary.csv', summary)
    pairs = []
    by_condition = {label: {row['seed']: row for row in rows if row['condition'] == label} for label, _, _, _ in CONDITIONS}
    for step in (1000, 3000, 10000, 30000):
        label = f'cp{step}_target'
        item = {'comparison': f'{label}-m0_target', 'n': 20}
        for metric in METRICS:
            diffs = [float(by_condition[label][seed][metric]) - float(by_condition['m0_target'][seed][metric]) for seed in SEEDS]
            item[metric + '_mean_diff'] = stats.mean(diffs)
            item[metric + '_sample_sd_diff'] = stats.stdev(diffs)
        pairs.append(item)
    save_csv(out / 'paired_differences.csv', pairs)
    for row in summary:
        print('CONDITION', json.dumps(row))
    for row in pairs:
        print('PAIRED', json.dumps(row))
if __name__ == '__main__':
    main()
