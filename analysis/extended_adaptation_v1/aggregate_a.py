"""Combine old and new 100-seed v2 evaluation summaries with paired differences."""
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
ROOT = artifact_dir('extended_adaptation_v1')
OUT = ROOT / 'aggregate'
OLD = artifact_dir('score_probe_v1') / 'analysis/a_seed_results.csv'
NEW = ROOT / 'evaluations'
MODELS = ('m0', 'cp10000', 'cp30000', '30k_c', '60k', '90k')
FIELDS = ('fall', 'episode_length', 'return', 'reward_per_step', 'mean_vx_body_mps', 'mean_vx_abs_tracking_error_mps')
PAIRS = (('90k', 'cp30000'), ('90k', 'm0'), ('30k_c', 'cp30000'))

def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def mean_sd(values):
    x = list(values)
    return (st.mean(x), st.stdev(x))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = {}
    for row in csv.DictReader(OLD.open()):
        if row['model'] in MODELS[:3]:
            key = (row['model'], int(row['seed']))
            assert key not in old
            old[key] = resolve_artifact_path(row['raw_summary'])
    results = {}
    seed_rows = []
    for model in MODELS:
        for seed in range(23501, 23601):
            path = old[model, seed] if model in MODELS[:3] else NEW / model / f'mppi_mass8_v10_seed{seed}' / 'summary.json'
            source = json.loads(path.read_text())
            assert source['seed'] == seed and source['warmup_termination_count'] == 0
            assert source['episode_length'] > 0 and source['status'] == 'evaluated'
            row = {'model': model, 'seed': seed, **{key: source[key] for key in FIELDS}, 'end_reason': source['end_reason'], 'termination_terms': source['termination_terms'], 'raw_summary': str(path.resolve())}
            if model not in MODELS[:3]:
                step_path = path.parent / 'evaluation_steps.csv'
                step_rows = list(csv.DictReader(step_path.open()))
                assert len(step_rows) == row['episode_length']
                assert abs(sum((float(x['reward']) for x in step_rows)) - row['return']) < 1e-08
            results[model, seed] = row
            seed_rows.append(row)
    assert len(seed_rows) == 600
    write('seed_results.csv', seed_rows)
    summary = []
    for model in MODELS:
        selected = [results[model, seed] for seed in range(23501, 23601)]
        item = {'model': model, 'seeds': 100, 'failure_count': sum((r['fall'] for r in selected))}
        for field in FIELDS:
            item[field + '_mean'], item[field + '_sample_sd'] = mean_sd((r[field] for r in selected))
        summary.append(item)
    write('condition_summary.csv', summary)
    paired, crosstabs = ([], [])
    for left, right in PAIRS:
        name = f'{left}-minus-{right}'
        left_only = right_only = both_fail = both_survive = 0
        for seed in range(23501, 23601):
            a, b = (results[left, seed], results[right, seed])
            paired.append({'comparison': name, 'seed': seed, **{field + '_difference': a[field] - b[field] for field in FIELDS}})
            if a['fall'] and b['fall']:
                both_fail += 1
            elif a['fall']:
                left_only += 1
            elif b['fall']:
                right_only += 1
            else:
                both_survive += 1
        crosstabs.append({'comparison': name, 'both_fail': both_fail, 'left_only_fail': left_only, 'right_only_fail': right_only, 'both_survive': both_survive})
    write('paired_seed_differences.csv', paired)
    write('failure_crosstabs.csv', crosstabs)
    pair_summary = []
    for left, right in PAIRS:
        name = f'{left}-minus-{right}'
        selected = [r for r in paired if r['comparison'] == name]
        item = {'comparison': name, 'seeds': 100}
        for field in FIELDS:
            item[field + '_mean_difference'], item[field + '_sample_sd_difference'] = mean_sd((r[field + '_difference'] for r in selected))
        pair_summary.append(item)
    write('paired_summary.csv', pair_summary)
    for row in summary:
        print('CONDITION', json.dumps(row))
    for row in pair_summary:
        print('PAIR', json.dumps(row))
    for row in crosstabs:
        print('CROSSTAB', json.dumps(row))
if __name__ == '__main__':
    main()
