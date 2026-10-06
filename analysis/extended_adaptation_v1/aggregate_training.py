"""Aggregate collection, training and uncontaminated independent-loss metadata."""
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

def write(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    collections = []
    for stage, first in (('60k', 33601), ('90k', 33701)):
        for seed in range(first, first + 6):
            path = ROOT / 'collections' / stage / f'seed_{seed}' / 'summary.json'
            s = json.loads(path.read_text())
            assert s['seed'] == seed and s['completed_control_steps'] == 5000
            assert s['warmup_terminations'] == 0
            collections.append({'stage': stage, 'seed': seed, 'split': 'heldout' if seed == first + 5 else 'train', 'control_steps': s['completed_control_steps'], 'falls': s['falls'], 'episodes': len(s['episodes']), 'warmup_steps_excluded': s['warmup_steps_excluded'], 'warmup_terminations': s['warmup_terminations'], 'duration_seconds': s['duration_seconds'], 'raw_summary': str(path.resolve())})
    write('collection_seeds.csv', collections)
    stages = []
    for stage in ('30k_c', '60k', '90k'):
        path = ROOT / 'training' / stage / 'summary.json'
        s = json.loads(path.read_text())
        split = json.loads((path.parent / 'data_split.json').read_text())
        assert s['frozen_parameter_changes'] == []
        assert s['max_epochs'] == 10
        stages.append({'stage': stage, 'cumulative_control_steps': {'30k_c': 30000, '60k': 60000, '90k': 90000}[stage], 'train_windows': s['cumulative_train_windows'], 'heldout_windows': s['cumulative_heldout_windows'], 'latest_heldout_windows': s['latest_heldout_windows'], 'max_updates': s['max_updates'], 'actual_updates': s['actual_updates'], 'selected_best_update': s['selected_best_update'], 'stop_reason': s['stop_reason'], 'initial_heldout_loss': s['initial_heldout_all_loss'], 'final_heldout_loss': s['heldout_all_loss'], 'latest_heldout_loss': s['heldout_latest_loss'], 'frozen_parameter_changes': len(s['frozen_parameter_changes']), 'dynamics_parameter_change_count': s['dynamics_parameter_change_count'], 'duration_seconds': s['duration_seconds'], 'checkpoint': s['checkpoint'], 'raw_summary': str(path.resolve()), 'split_metadata': str((path.parent / 'data_split.json').resolve())})
    write('training_stages.csv', stages)
    losses = list(csv.DictReader((ROOT / 'independent_losses/c_per_seed_losses.csv').open()))
    clean = []
    for stage in ('30k_c', '60k', '90k'):
        for policy in ('m0', 'cp10000'):
            selected = [r for r in losses if r['model'] == stage and r['collection_policy'] == policy and (policy == 'm0' or int(r['seed']) != 35001)]
            n = len(selected)
            assert n == (10 if policy == 'm0' else 9)
            item = {'model': stage, 'collection_policy': policy, 'seeds': n, 'excluded_heldout_seed': 35001 if policy == 'cp10000' else '', 'windows': sum((int(r['windows']) for r in selected))}
            for key in ('one_step_latent_mse', 'full_25_step_latent_mse'):
                values = [float(r[key]) for r in selected]
                weights = [int(r['windows']) for r in selected]
                item[key + '_pooled_window_mean'] = sum((v * w for v, w in zip(values, weights))) / sum(weights)
                item[key + '_seed_mean'] = st.mean(values)
                item[key + '_seed_sample_sd'] = st.stdev(values)
            clean.append(item)
    write('independent_loss_clean.csv', clean)
    for r in stages:
        print('TRAIN', json.dumps(r))
    for r in clean:
        print('LOSS_CLEAN', json.dumps(r))
if __name__ == '__main__':
    main()
