"""One-step and full-25-step latent MSE on two independent policy data sets."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.65')
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
ROOT = artifact_dir('extended_adaptation_v1')
SIMDIST = simdist_root()
COLLECTIONS = artifact_dir('score_probe_v1') / 'validation_collections'
OUT = ROOT / 'independent_losses'
H = T = 25
MODELS = {**{stage: ROOT / 'training' / stage / 'checkpoint' for stage in ('30k_c', '60k', '90k')}}

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def read_seed(source, seed):
    path = COLLECTIONS / source / f'seed_{seed}'
    summary = json.loads((path / 'summary.json').read_text())
    assert summary['seed'] == seed and summary['completed_control_steps'] == 500
    episodes = []
    with h5py.File(path / 'rollout.h5', 'r') as f:
        for name in sorted(f.keys()):
            episodes.append({key: np.asarray(f[name][key], np.float32) for key in ('proprio_obs', 'extero_obs', 'actions', 'commands', 'reward')})
    assert sum((len(ep['actions']) for ep in episodes)) == 500
    refs = [(ep_index, start) for ep_index, ep in enumerate(episodes) for start in range(max(0, len(ep['actions']) - H - T))]
    assert refs, (source, seed, [len(ep['actions']) for ep in episodes])
    return (episodes, refs, summary)

def make_batch(episodes, refs):
    parts = {'model_in': {key: [] for key in ('proprio_obs_hist', 'extero_obs', 'acts_hist', 'fut_acts', 'fut_cmds')}, 'labels': {key: [] for key in ('proprio_obs', 'extero_obs')}}
    for ep_index, start in refs:
        ep = episodes[ep_index]
        t, u = (start + H, start + H + T)
        x = parts['model_in']
        y = parts['labels']
        x['proprio_obs_hist'].append(ep['proprio_obs'][start:t + 1])
        x['extero_obs'].append(ep['extero_obs'][t])
        x['acts_hist'].append(ep['actions'][start:t])
        x['fut_acts'].append(ep['actions'][t:u])
        x['fut_cmds'].append(ep['commands'][t:u + 1])
        y['proprio_obs'].append(ep['proprio_obs'][t + 1:u + 1])
        y['extero_obs'].append(ep['extero_obs'][t + 1:u + 1])
    return {part: {key: np.stack(values) for key, values in group.items()} for part, group in parts.items()}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    collections = {source: {seed: read_seed(source, seed) for seed in range(35001, 35011)} for source in ('m0', 'cp10000')}
    per_seed = []
    for model_name, checkpoint in MODELS.items():
        model, _, checkpoint_step = model_utils.load_model_from_ckpt(str(checkpoint.resolve()), None)

        @nnx.jit
        def loss_step(model, x, future_obs):
            y_pred = model(x, deterministic=True)
            scaled = jax.lax.stop_gradient(model.get_scaler().scale(future_obs))
            target = jax.lax.stop_gradient(model.encode_latent(scaled['proprio_obs'], scaled['extero_obs'], deterministic=True))
            squared = (y_pred['latents'] - target) ** 2
            return (jnp.mean(squared[:, 0, :]), jnp.mean(squared))
        for source, seeds in collections.items():
            for seed, (episodes, refs, summary) in seeds.items():
                total_first = total_full = 0.0
                for start in range(0, len(refs), 32):
                    chosen = refs[start:start + 32]
                    batch = model_utils.dataset_batch_to_jax(make_batch(episodes, chosen))
                    first, full = loss_step(model, batch['model_in'], batch['labels'])
                    total_first += float(first) * len(chosen)
                    total_full += float(full) * len(chosen)
                row = {'model': model_name, 'checkpoint': str(checkpoint.resolve()), 'checkpoint_step': checkpoint_step, 'collection_policy': source, 'seed': seed, 'control_steps': 500, 'windows': len(refs), 'falls': summary['falls'], 'warmup_terminations': summary['warmup_terminations'], 'one_step_latent_mse': total_first / len(refs), 'full_25_step_latent_mse': total_full / len(refs), 'raw_rollout': str((COLLECTIONS / source / f'seed_{seed}' / 'rollout.h5').resolve())}
                per_seed.append(row)
                print('LOSS', json.dumps(row), flush=True)
    write_csv(OUT / 'c_per_seed_losses.csv', per_seed)
    summary_rows = []
    for model_name in MODELS:
        for source in ('m0', 'cp10000'):
            selected = [r for r in per_seed if r['model'] == model_name and r['collection_policy'] == source]
            assert len(selected) == 10
            weights = [r['windows'] for r in selected]
            item = {'model': model_name, 'collection_policy': source, 'seeds': len(selected), 'control_steps': 5000, 'total_windows': sum(weights), 'collection_falls': sum((r['falls'] for r in selected))}
            item['warmup_terminations'] = sum((r['warmup_terminations'] for r in selected))
            for key in ('one_step_latent_mse', 'full_25_step_latent_mse'):
                values = [r[key] for r in selected]
                item[key + '_pooled_window_mean'] = sum((v * w for v, w in zip(values, weights))) / sum(weights)
                item[key + '_seed_mean'] = st.mean(values)
                item[key + '_seed_sample_sd'] = st.stdev(values)
            summary_rows.append(item)
    write_csv(OUT / 'c_loss_summary.csv', summary_rows)
    for row in summary_rows:
        print('SUMMARY', json.dumps(row), flush=True)
if __name__ == '__main__':
    main()
