"""Merge the previous 20 v2 episodes with 80 new episodes per condition."""
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
ROOT = artifact_dir('score_probe_v1')
OLD = artifact_dir('adaptation_curve_v1') / 'aggregate/seed_results.csv'
NEW = ROOT / 'evaluations'
OUT = ROOT / 'analysis'
LABELS = ('m0', 'cp10000', 'cp30000', 'base')
OLD_LABEL = {'m0': 'm0_target', 'cp10000': 'cp10000_target', 'cp30000': 'cp30000_target', 'base': 'base_target'}
METRICS = ('fall', 'episode_length', 'return', 'reward_per_step')
PAIRS = (('cp10000', 'm0'), ('cp30000', 'm0'), ('cp30000', 'cp10000'), ('cp10000', 'base'))

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    old_rows = list(csv.DictReader(OLD.open()))
    old_lookup = {(row['condition'], int(row['seed'])): row['raw_summary'] for row in old_rows}
    rows = []
    for label in LABELS:
        controller = 'base' if label == 'base' else 'mppi'
        for seed in range(23501, 23601):
            if seed <= 23520:
                summary_path = resolve_artifact_path(old_lookup[OLD_LABEL[label], seed])
            else:
                summary_path = NEW / label / f'{controller}_mass8_v10_seed{seed}' / 'summary.json'
            assert summary_path.exists(), summary_path
            s = json.loads(summary_path.read_text())
            assert s['status'] == 'evaluated' and s['seed'] == seed
            assert s['controller'] == controller and s['mass_added_kg'] == 8.0
            assert s['forward_target_mps'] == 1.0 and s['warmup_steps'] == 25
            rows.append({'model': label, 'seed': seed, **{key: s[key] for key in METRICS}, 'end_reason': s['end_reason'], 'termination_terms': s['termination_terms'], 'warmup_terminations': s['warmup_termination_count'], 'score_snapshot_count': s.get('score_snapshot_count', 0), 'raw_summary': str(summary_path.resolve())})
    assert len(rows) == 400 and len({(r['model'], r['seed']) for r in rows}) == 400
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'a_seed_results.csv', rows)
    conditions = []
    for label in LABELS:
        selected = [r for r in rows if r['model'] == label]
        item = {'model': label, 'n': len(selected), 'failure_count': sum((int(r['fall']) for r in selected)), 'warmup_terminations': sum((int(r['warmup_terminations']) for r in selected))}
        for key in METRICS:
            values = [float(r[key]) for r in selected]
            item[key + '_mean'] = st.mean(values)
            item[key + '_sample_sd'] = st.stdev(values)
        conditions.append(item)
    write_csv(OUT / 'a_conditions.csv', conditions)
    by_model = {label: {int(r['seed']): r for r in rows if r['model'] == label} for label in LABELS}
    paired = []
    crosses = []
    for left, right in PAIRS:
        row = {'comparison': f'{left}-{right}', 'n': 100}
        for key in METRICS:
            differences = [float(by_model[left][seed][key]) - float(by_model[right][seed][key]) for seed in range(23501, 23601)]
            row[key + '_mean_diff'] = st.mean(differences)
            row[key + '_sample_sd_diff'] = st.stdev(differences)
        paired.append(row)
        states = [(int(by_model[left][seed]['fall']), int(by_model[right][seed]['fall'])) for seed in range(23501, 23601)]
        crosses.append({'comparison': f'{left}-{right}', 'both_fail': states.count((1, 1)), 'left_only_fail': states.count((1, 0)), 'right_only_fail': states.count((0, 1)), 'both_survive': states.count((0, 0))})
    write_csv(OUT / 'a_paired.csv', paired)
    write_csv(OUT / 'a_failure_crosstabs.csv', crosses)
    for row in conditions + paired + crosses:
        print(json.dumps(row), flush=True)
if __name__ == '__main__':
    main()
