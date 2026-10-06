"""Fresh-process released dynamics-only Trainer invocation plus frozen-state audit."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import argparse
import datetime
import os
import json
from pathlib import Path
from common import ROOT, REPO, SOURCE, config, route_paths, write_json, hashes

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--steps', required=True, type=int)
    p.add_argument('--training-data-ratio', type=float, default=None, help='Optional explicit Trainer train fraction. 1.0 is used by the fixed-window T1 control so all manifest windows are optimizer-eligible.')
    p.add_argument('--allowed-training-seeds', default='2101,2102,2103')
    p.add_argument('--allowed-collection-seeds', default='1101,1102,1103')
    p.add_argument('--log-dir', default=None, help='Fresh optional TensorBoard/raw artifact directory.')
    p.add_argument('--heldout-name', default=None, help='Fixed external evaluation dataset name for curve logging.')
    p.add_argument('--heldout-dataset-root', default=None)
    p.add_argument('--training-dataset-root', default=None, help='Read-only root for a frozen external training dataset; model outputs remain in this run root.')
    p.add_argument('--logging-eval-interval', type=int, default=None)
    p.add_argument('--smoke-single-process-loader', action='store_true', help='Smoke-only technical switch to avoid JAX/fork deadlock; never use for a scientific run.')
    args = p.parse_args()
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.65'
    from simdist.utils.jax import configure_jax_compilation_cache
    configure_jax_compilation_cache()
    import numpy as np
    import random
    import torch
    import flax.nnx as nnx
    import jax
    from simdist.modeling import trainer
    from simdist.utils.model import load_model_from_ckpt, create_param_filter
    from simdist.utils import paths
    route_paths()
    data_root = resolve_artifact_path(args.training_dataset_root or str(ROOT / 'datasets')).resolve()
    paths._PATHS['ALL_DATASETS'] = str(data_root)
    paths._PATHS['REAL_DATASETS'] = str(data_root)
    cfg = config('finetune_model')
    cfg['finetune'] = True
    cfg['run_name'] = args.run
    cfg['data']['dataset_name'] = args.name
    cfg['training']['seed'] = args.seed
    cfg['training']['max_steps'] = args.steps
    if args.training_data_ratio is not None:
        assert 0.0 < args.training_data_ratio <= 1.0
        cfg['training']['training_data_ratio'] = args.training_data_ratio
    cfg['checkpoint']['resume_checkpoint'] = str(SOURCE)
    if args.smoke_single_process_loader:
        assert args.steps <= 2 and args.seed == 42 and (args.name.startswith('smoke_') or args.name.startswith('01_smoke_'))
        cfg['data']['num_train_workers'] = 0
        cfg['data']['num_test_workers'] = 0
    model_dir = ROOT / 'models' / args.run
    assert not model_dir.exists(), 'Run directory must be fresh'
    manifest = json.loads((data_root / args.name / 'dataset_manifest.json').read_text())
    assert manifest['role'] in ['collection', 'smoke'], 'D_B_eval contamination'
    assert hashes(manifest['processed_dir']) == manifest['processed_hashes']
    if manifest['role'] != 'smoke':
        allowed_training = {int(x) for x in args.allowed_training_seeds.split(',')}
        allowed_collection = {int(x) for x in args.allowed_collection_seeds.split(',')}
        assert args.seed in allowed_training
        assert set(manifest['seeds']).issubset(allowed_collection)
    if args.log_dir is not None:
        assert args.heldout_name is not None, '--heldout-name is required when --log-dir is used'
        heldout_root = os.path.abspath(args.heldout_dataset_root or str(data_root))
        heldout_manifest = json.loads((resolve_artifact_path(heldout_root) / args.heldout_name / 'dataset_manifest.json').read_text())
        assert heldout_manifest['role'] in ['eval', 'smoke'], 'Held-out logger dataset must be eval/smoke, never collection'
        cfg['instrumentation'] = {'enabled': True, 'run_dir': str(resolve_artifact_path(args.log_dir).resolve()), 'heldout_dataset_name': args.heldout_name, 'heldout_dataset_root': heldout_root, 'eval_interval': args.logging_eval_interval or cfg['training']['eval_interval'], 'metadata': {'timestamp_requested_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': ' '.join(__import__('sys').argv), 'config_path': str(simdist_root() / 'config/finetune_model.yaml'), 'seed': args.seed, 'shift_family': manifest.get('intervention_family') or ('kp' if manifest.get('kp') is not None else 'added_mass' if manifest.get('added_mass_kg') else 'friction'), 'shift_value': manifest.get('kp') if manifest.get('intervention_family') == 'kp' else manifest.get('added_mass_kg') or manifest.get('friction'), 'source_condition': {'checkpoint': str(SOURCE), 'step': 367004}, 'target_condition': {k: manifest.get(k) for k in ['intervention_family', 'friction', 'added_mass_kg', 'kp', 'kd', 'fixed_kp', 'fixed_kd']}, 'checkpoint_path': str(SOURCE), 'dataset_identifier': args.name, 'heldout_dataset_identifier': args.heldout_name, 'adaptation_hyperparameters': cfg['training']}}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    before, _, initial_step = load_model_from_ckpt(str(SOURCE), 367004)
    frozen_before = nnx.state(before, nnx.Not(create_param_filter(['dynamics']))).to_pure_dict()
    params_before = nnx.state(before, create_param_filter(['dynamics'])).to_pure_dict()

    def as_numpy(x):
        if hasattr(x, 'dtype') and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x = jax.random.key_data(x)
        return np.asarray(x)
    frozen_before = jax.tree.map(lambda x: as_numpy(x).copy(), frozen_before)
    params_before = jax.tree.map(lambda x: np.asarray(x).copy(), params_before)

    def flatten(tree, prefix=()):
        if isinstance(tree, dict):
            return {path: value for k, v in tree.items() for path, value in flatten(v, prefix + (str(k),)).items()}
        return {prefix: as_numpy(tree)}
    del before
    audit_dir = ROOT / 'adaptation' / args.run
    audit_dir.mkdir(parents=True, exist_ok=False)
    write_json(audit_dir / 'invocation.json', {'config': cfg, 'dataset_manifest': manifest, 'initial_step': initial_step, 'optimizer_local_step': 0, 'scheduler_local_step': 0, 'optimizer': 'fresh optax.adam', 'source_hashes': hashes(SOURCE)})
    trainer.train(cfg)
    actual_updates = None
    if args.log_dir is not None:
        completion = json.loads((resolve_artifact_path(args.log_dir) / 'raw' / 'trainer_completion.json').read_text())
        actual_updates = int(completion['actual_optimizer_updates'])
        assert actual_updates == args.steps, f'Actual optimizer counter mismatch: {actual_updates} versus {args.steps}'
    after, _, final_step = load_model_from_ckpt(str(model_dir))
    updates = final_step - initial_step
    frozen_after = flatten(nnx.state(after, nnx.Not(create_param_filter(['dynamics']))).to_pure_dict())
    changed = []
    rng_changes = []
    for path, a in flatten(frozen_before).items():
        b = frozen_after[path]
        if not np.array_equal(a, b):
            if any(('rng' in v.lower() for v in path)):
                rng_changes.append('/'.join(path))
            else:
                changed.append('/'.join(path))
    assert not changed, f'Frozen state changed: {changed}'
    dynamics_after = flatten(nnx.state(after, create_param_filter(['dynamics'])).to_pure_dict())
    dynamics_changes = ['/'.join(path) for path, a in flatten(params_before).items() if not np.array_equal(a, dynamics_after[path])]
    assert dynamics_changes, 'No dynamics weight changes'
    assert updates == args.steps, f'Budget not reached: {updates} versus {args.steps}'
    assert hashes(manifest['processed_dir']) == manifest['processed_hashes']
    write_json(audit_dir / 'result.json', {'initial_step': initial_step, 'final_step': final_step, 'optimizer_updates': updates, 'actual_optimizer_updates': actual_updates if actual_updates is not None else updates, 'scheduler_local_step': updates, 'frozen_changes': changed, 'rng_counter_changes': rng_changes, 'dynamics_changes': dynamics_changes, 'source_scaler_unchanged': True, 'fresh_process_pid': os.getpid(), 'training_data_ratio': cfg['training']['training_data_ratio'], 'train_windows': int(np.ceil(manifest['eligible_windows'] * cfg['training']['training_data_ratio'])), 'checkpoint': str(model_dir), 'checkpoint_hashes': hashes(model_dir)})
    print('ADAPTATION_COMPLETE', updates, flush=True)
if __name__ == '__main__':
    main()
