"""Measure one-step and full-25-step latent MSE on named rollout datasets."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import argparse
import csv
import json
import statistics as st
from pathlib import Path
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from simdist.utils import model as model_utils
parser = argparse.ArgumentParser()
parser.add_argument('--model', action='append', required=True, help='label=checkpoint')
parser.add_argument('--dataset', action='append', required=True, help='label=H5 file or directory')
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
H = T = 25

def pairs(items):
    result = {}
    for item in items:
        label, value = item.split('=', 1)
        result[label] = resolve_artifact_path(value).resolve()
    return result

def rollout_files(path):
    if path.is_file():
        return [path]
    return sorted(path.rglob('rollout.h5'))

def read_episodes(path):
    episodes = []
    with h5py.File(path, 'r') as stream:
        for name in sorted(stream.keys()):
            group = stream[name]
            episodes.append({key: np.asarray(group[key], np.float32) for key in ('proprio_obs', 'extero_obs', 'actions', 'commands')})
    return episodes

def batch_from_refs(episodes, refs):
    parts = {'model_in': {key: [] for key in ('proprio_obs_hist', 'extero_obs', 'acts_hist', 'fut_acts', 'fut_cmds')}, 'labels': {key: [] for key in ('proprio_obs', 'extero_obs')}}
    for ep_index, start in refs:
        ep = episodes[ep_index]
        t, end = (start + H, start + H + T)
        x, y = (parts['model_in'], parts['labels'])
        x['proprio_obs_hist'].append(ep['proprio_obs'][start:t + 1])
        x['extero_obs'].append(ep['extero_obs'][t])
        x['acts_hist'].append(ep['actions'][start:t])
        x['fut_acts'].append(ep['actions'][t:end])
        x['fut_cmds'].append(ep['commands'][t:end + 1])
        y['proprio_obs'].append(ep['proprio_obs'][t + 1:end + 1])
        y['extero_obs'].append(ep['extero_obs'][t + 1:end + 1])
    return {part: {key: np.stack(values) for key, values in group.items()} for part, group in parts.items()}

def main():
    models = pairs(args.model)
    datasets = pairs(args.dataset)
    rows = []
    for model_label, checkpoint in models.items():
        model, _, step = model_utils.load_model_from_ckpt(str(checkpoint), None)

        @nnx.jit
        def loss_step(model, x, future_obs):
            prediction = model(x, deterministic=True)
            scaled = jax.lax.stop_gradient(model.get_scaler().scale(future_obs))
            target = jax.lax.stop_gradient(model.encode_latent(scaled['proprio_obs'], scaled['extero_obs'], deterministic=True))
            squared = (prediction['latents'] - target) ** 2
            return (jnp.mean(squared[:, 0, :]), jnp.mean(squared))
        for dataset_label, root in datasets.items():
            files = rollout_files(root)
            assert files, f'no rollout.h5 found under {root}'
            for file in files:
                episodes = read_episodes(file)
                refs = [(index, start) for index, ep in enumerate(episodes) for start in range(max(0, len(ep['actions']) - H - T))]
                assert refs, f'no valid 50-step latent windows: {file}'
                first_sum = full_sum = 0.0
                for offset in range(0, len(refs), 32):
                    selected = refs[offset:offset + 32]
                    batch = model_utils.dataset_batch_to_jax(batch_from_refs(episodes, selected))
                    first, full = loss_step(model, batch['model_in'], batch['labels'])
                    first_sum += float(first) * len(selected)
                    full_sum += float(full) * len(selected)
                rows.append({'model': model_label, 'checkpoint': str(checkpoint), 'checkpoint_step': step, 'dataset': dataset_label, 'rollout': str(file.resolve()), 'windows': len(refs), 'one_step_latent_mse': first_sum / len(refs), 'full_25_step_latent_mse': full_sum / len(refs)})
                print('LOSS', json.dumps(rows[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for model_label in models:
        for dataset_label in datasets:
            chosen = [row for row in rows if row['model'] == model_label and row['dataset'] == dataset_label]
            for key in ('one_step_latent_mse', 'full_25_step_latent_mse'):
                values = [row[key] for row in chosen]
                summary.append({'model': model_label, 'dataset': dataset_label, 'rollouts': len(values), 'windows': sum((row['windows'] for row in chosen)), 'metric': key, 'seed_mean': st.mean(values), 'seed_sample_sd': st.stdev(values) if len(values) > 1 else 0.0, 'pooled_window_mean': sum((row[key] * row['windows'] for row in chosen)) / sum((row['windows'] for row in chosen))})
    with args.out.with_name(args.out.stem + '_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print('SUMMARY', json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
