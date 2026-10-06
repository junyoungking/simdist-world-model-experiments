"""Paired 100-seed comparison of the independent 30k adaptation chain."""
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
ROOT = artifact_dir('followup_v1')
REPO = PORTFOLIO_ROOT
OUT = ROOT / 'chain/aggregate'
OLD = result_path('extended_adaptation_v1/aggregate/seed_results.csv')
NEW = ROOT / 'chain/evaluation'
FIELDS = ('fall', 'episode_length', 'return', 'reward_per_step', 'mean_vx_body_mps', 'mean_vx_abs_tracking_error_mps')

def write(name, records):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

def mean_sd(values):
    values = list(values)
    return (st.mean(values), st.stdev(values))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = {int(r['seed']): resolve_artifact_path(r['raw_summary']) for r in csv.DictReader(OLD.open()) if r['model'] == 'cp30000'}
    assert len(old) == 100
    rows, indexed = ([], {})
    for model in ('old_30k', 'new_30k'):
        for seed in range(23501, 23601):
            path = old[seed] if model == 'old_30k' else NEW / f'mppi_mass8_v10_seed{seed}/summary.json'
            s = json.loads(path.read_text())
            assert s['seed'] == seed and s['status'] == 'evaluated'
            assert s['warmup_termination_count'] == 0
            row = {'model': model, 'seed': seed}
            row.update({key: s[key] for key in FIELDS})
            row.update({'end_reason': s['end_reason'], 'raw_summary': str(path.resolve())})
            rows.append(row)
            indexed[model, seed] = row
    write('seed_results.csv', rows)
    summary = []
    for model in ('old_30k', 'new_30k'):
        selected = [indexed[model, seed] for seed in range(23501, 23601)]
        row = {'model': model, 'seeds': 100, 'failures': sum((r['fall'] for r in selected))}
        for key in FIELDS:
            row[key + '_mean'], row[key + '_sample_sd'] = mean_sd((r[key] for r in selected))
        summary.append(row)
    write('condition_summary.csv', summary)
    paired = []
    for seed in range(23501, 23601):
        left, right = (indexed['new_30k', seed], indexed['old_30k', seed])
        row = {'seed': seed, 'new_failed': left['fall'], 'old_failed': right['fall']}
        row.update({key + '_difference': left[key] - right[key] for key in FIELDS})
        paired.append(row)
    write('paired_seed_differences.csv', paired)
    row = {'seeds': 100, 'both_fail': sum((r['new_failed'] and r['old_failed'] for r in paired)), 'new_only_fail': sum((r['new_failed'] and (not r['old_failed']) for r in paired)), 'old_only_fail': sum((r['old_failed'] and (not r['new_failed']) for r in paired)), 'both_survive': sum((not r['new_failed'] and (not r['old_failed']) for r in paired))}
    for key in FIELDS:
        row[key + '_mean_difference'], row[key + '_sample_sd_difference'] = mean_sd((r[key + '_difference'] for r in paired))
    write('paired_summary.csv', [row])
    for item in summary:
        print('CONDITION', json.dumps(item), flush=True)
    print('PAIRED', json.dumps(row), flush=True)
if __name__ == '__main__':
    main()
