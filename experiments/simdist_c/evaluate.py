"""Deterministic external evaluator, independent of Trainer's internal split."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import argparse
import json
import os
from pathlib import Path
from common import ROOT, SOURCE, config, route_paths, write_json, hashes, close_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--checkpoint', default=str(SOURCE))
    p.add_argument('--step', type=int, default=None, help='Explicit saved checkpoint step; default is latest.')
    p.add_argument('--output', required=True)
    p.add_argument('--dataset-root', default=None, help='Read-only dataset root; defaults to this phase output datasets directory.')
    p.add_argument('--allow-training-view', action='store_true', help='Permit deterministic diagnostics on a frozen training view; never marks it external.')
    args = p.parse_args()
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.5'
    from simdist.utils.jax import configure_jax_compilation_cache
    configure_jax_compilation_cache()
    import numpy as np
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from simdist.utils.model import load_model_from_ckpt
    from simdist.data.dataset import get_dataset
    from simdist.modeling.losses import WorldModelLoss
    route_paths()
    if args.dataset_root is not None:
        from simdist.utils import paths
        paths._PATHS['ALL_DATASETS'] = str(resolve_artifact_path(args.dataset_root).resolve())
        paths._PATHS['REAL_DATASETS'] = str(resolve_artifact_path(args.dataset_root).resolve())
    out = resolve_artifact_path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    model, cfg, step = load_model_from_ckpt(args.checkpoint, args.step)
    cfg['loss'] = config('finetune_model')['loss']
    loss_fn = WorldModelLoss(cfg)

    @nnx.jit
    def predict(model, x, y):
        prediction = model(x, deterministic=True)['latents']
        scaled = model.get_scaler().scale(y)
        target = model.encode_latent(scaled['proprio_obs'], scaled['extero_obs'], deterministic=True)
        official, _ = loss_fn.compute_losses(model, x, y, deterministic=True)
        return (prediction, target, official['latent_dynamics'])
    results = []
    for name in args.names:
        folder = (resolve_artifact_path(args.dataset_root).resolve() if args.dataset_root is not None else ROOT / 'datasets') / name
        manifest = json.loads((folder / 'dataset_manifest.json').read_text())
        assert manifest['role'] in ['eval', 'smoke'] or (args.allow_training_view and manifest['role'] == 'training_view'), 'Collection data cannot be external evaluation'
        assert hashes(manifest['processed_dir']) == manifest['processed_hashes']
        cfg['data'] = {'dataset_name': name}
        dataset = get_dataset(cfg)
        dataset.eval()
        preds, targets, window_indices = ([], [], [])
        for start in range(0, len(dataset), 32):
            items = [dataset[i] for i in range(start, min(start + 32, len(dataset)))]
            x = jax.tree.map(lambda *a: jnp.asarray(np.stack(a)), *[i['model_in'] for i in items])
            y = jax.tree.map(lambda *a: jnp.asarray(np.stack(a)), *[i['labels'] for i in items])
            assert all((np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves((x, y))))
            prediction, target, official = predict(model, x, y)
            prediction, target = (np.asarray(prediction), np.asarray(target))
            assert np.isfinite(prediction).all() and np.isfinite(target).all()
            measured = np.mean(np.square(prediction - target))
            np.testing.assert_allclose(measured, float(official), rtol=2e-05, atol=1e-07)
            preds.append(prediction)
            targets.append(target)
            window_indices.extend((int(i['metadata']['t_H']) for i in items))
        prediction, target = (np.concatenate(preds), np.concatenate(targets))
        error = np.square(prediction.astype(np.float64) - target.astype(np.float64))
        np.savez_compressed(out / f'{name}.npz', predictions=prediction, targets=target, start_idxs=window_indices, per_window_mse=error.mean(axis=(1, 2)))
        results.append({'name': name, 'seed': manifest['seed'], 'windows': len(dataset), 'latent_elements': int(error.size), 'sse': float(error.sum()), 'latent_mse': float(error.mean()), 'per_horizon_mse': error.mean(axis=(0, 2)).tolist(), 'dataset_manifest': manifest})
        close_dataset(dataset)
        assert hashes(manifest['processed_dir']) == manifest['processed_hashes']
    losses = [r['latent_mse'] for r in results]
    write_json(out / 'metrics.json', {'checkpoint': args.checkpoint, 'step': step, 'checkpoint_hashes': hashes(resolve_artifact_path(args.checkpoint)), 'per_seed': results, 'mean_seed_loss': float(np.mean(losses)), 'sd_seed_loss': float(np.std(losses, ddof=1)) if len(losses) > 1 else None, 'pooled_window_loss': sum((r['sse'] for r in results)) / sum((r['latent_elements'] for r in results)), 'aggregation': 'Qualification uses equally weighted seed means; pooled window loss is secondary.', 'metric': 'Released deterministic latent MSE; mean over windows, 25 future steps and 64 latents', 'external': not args.allow_training_view, 'training': bool(args.allow_training_view)})
    print('EVALUATION_COMPLETE', losses, flush=True)
if __name__ == '__main__':
    main()
