"""Dynamics-only SimDist fine-tuning with a fixed held-out early stop."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.65')
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
import jax
import numpy as np
import orbax.checkpoint as ocp
import optax
import yaml
import flax.nnx as nnx
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from simdist.modeling import losses
from simdist.utils import model as model_utils
from curve_data import CurveData
ap = argparse.ArgumentParser()
ap.add_argument('--previous', type=Path, required=True)
ap.add_argument('--previous-step', type=int)
ap.add_argument('--rollouts', type=Path, nargs='+', required=True)
ap.add_argument('--seed', type=int, default=34501)
ap.add_argument('--max-updates', type=int, default=4000)
ap.add_argument('--eval-every', type=int, default=100)
ap.add_argument('--patience', type=int, default=5)
ap.add_argument('--min-updates', type=int, default=500)
ap.add_argument('--relative-improvement', type=float, default=0.001)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()

def flatten(tree, prefix=()):
    if isinstance(tree, dict):
        result = {}
        for key, value in tree.items():
            result.update(flatten(value, prefix + (str(key),)))
        return result
    return {'/'.join(prefix): np.asarray(tree)}

def main():
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model, model_cfg, previous_step = model_utils.load_model_from_ckpt(str(args.previous.resolve()), args.previous_step)
    with initialize_config_dir(config_dir=str(simdist_root() / 'config'), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name='finetune_model'), resolve=True)
    cfg['finetune'] = True
    cfg['model'] = copy.deepcopy(model_cfg['model'])
    cfg['system'] = copy.deepcopy(model_cfg['system'])
    cfg['scaler_params_struct'] = copy.deepcopy(model_cfg['scaler_params_struct'])
    cfg['training']['seed'] = args.seed
    cfg['training']['training_data_ratio'] = 1.0
    assert cfg['heads'] == ['dynamics']
    assert cfg['loss']['weights'] == {'latent_dynamics': 1.0, 'reward': 0.0, 'value': 0.0, 'action': 0.0}
    assert cfg['training']['batch_size'] == 32
    data = CurveData([str(path.resolve()) for path in args.rollouts], model_cfg)
    (args.out / 'data_split.json').write_text(json.dumps({'rollouts': [str(path.resolve()) for path in args.rollouts], 'intervals': data.intervals, 'cumulative_train_windows': len(data.train), 'cumulative_heldout_windows': len(data.heldout), 'latest_heldout_windows': len(data.latest), 'split_rule': 'last 10% control steps of each interval held out; windows spanning split excluded; warm-up rows absent from rollouts'}, indent=2) + '\n')
    filt = model_utils.create_param_filter(cfg['heads'])
    frozen_filter = nnx.All(nnx.Param, nnx.Not(filt))
    frozen_before = flatten(nnx.state(model, frozen_filter).to_pure_dict())
    dynamics_before = flatten(nnx.state(model, filt).to_pure_dict())
    loss_fn = losses.get_loss(cfg)
    lr = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=cfg['training']['learning_rate'], warmup_steps=cfg['training']['warmup_steps'], decay_steps=cfg['training']['decay_steps'], end_value=cfg['training']['end_learning_rate'])
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=filt)
    diff_state = nnx.DiffState(0, filt)

    @nnx.jit
    def train_step(model, optimizer, x, y):
        grad_fn = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)
        (loss, _), grads = grad_fn(model, x, y, False)
        optimizer.update(grads)
        return loss

    @nnx.jit
    def eval_step(model, x, y):
        return loss_fn(model, x, y, True)[0]

    def evaluate(refs):
        total = 0.0
        count = 0
        for start in range(0, len(refs), cfg['training']['batch_size']):
            selected = refs[start:start + cfg['training']['batch_size']]
            batch = model_utils.dataset_batch_to_jax(data.batch(selected, augment=False))
            value = float(eval_step(model, batch['model_in'], batch['labels']))
            assert np.isfinite(value)
            total += value * len(selected)
            count += len(selected)
        return total / count

    def snapshot():
        return jax.tree.map(lambda x: x, nnx.state(model).to_pure_dict())
    initial_all = evaluate(data.heldout)
    initial_latest = evaluate(data.latest)
    best_loss = initial_all
    best_update = 0
    best_pure = snapshot()
    significant_anchor = initial_all
    stale = 0
    curve = [{'update': 0, 'train_loss_recent': '', 'heldout_all_loss': initial_all, 'heldout_latest_loss': initial_latest, 'learning_rate': 0.0, 'best_heldout_loss': best_loss, 'stale_evaluations': 0}]
    order = np.random.permutation(len(data.train))
    cursor = 0
    train_values = []
    stop_reason = 'update_cap'
    actual_updates = 0
    for update in range(1, args.max_updates + 1):
        if cursor + 32 > len(order):
            order = np.random.permutation(len(data.train))
            cursor = 0
        indices = order[cursor:cursor + 32]
        cursor += 32
        batch = model_utils.dataset_batch_to_jax(data.batch([data.train[int(i)] for i in indices], augment=True))
        value = float(train_step(model, optimizer, batch['model_in'], batch['labels']))
        assert np.isfinite(value)
        train_values.append(value)
        actual_updates = update
        if update % args.eval_every == 0 or update == args.max_updates:
            heldout_all = evaluate(data.heldout)
            heldout_latest = evaluate(data.latest)
            if heldout_all < best_loss:
                best_loss = heldout_all
                best_update = update
                best_pure = snapshot()
            if heldout_all <= significant_anchor * (1.0 - args.relative_improvement):
                significant_anchor = heldout_all
                stale = 0
            else:
                stale += 1
            row = {'update': update, 'train_loss_recent': float(np.mean(train_values)), 'heldout_all_loss': heldout_all, 'heldout_latest_loss': heldout_latest, 'learning_rate': float(lr(update - 1)), 'best_heldout_loss': best_loss, 'stale_evaluations': stale}
            curve.append(row)
            print('TRAIN_PROGRESS', json.dumps(row), flush=True)
            train_values.clear()
            if update >= args.min_updates and stale >= args.patience:
                stop_reason = 'heldout_plateau'
                break
    with (args.out / 'training_curve.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(best_pure)
    model = nnx.merge(graphdef, state)
    final_all = evaluate(data.heldout)
    final_latest = evaluate(data.latest)
    assert abs(final_all - best_loss) < 1e-06
    frozen_after = flatten(nnx.state(model, frozen_filter).to_pure_dict())
    frozen_changes = [key for key in frozen_before if not np.array_equal(frozen_before[key], frozen_after[key])]
    dynamics_after = flatten(nnx.state(model, filt).to_pure_dict())
    dynamics_changes = [key for key in dynamics_before if not np.array_equal(dynamics_before[key], dynamics_after[key])]
    assert not frozen_changes
    assert best_update == 0 or dynamics_changes
    checkpoint = (args.out / 'checkpoint').resolve()
    checkpoint.mkdir()
    with (checkpoint / 'model_config.yaml').open('w') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    checkpoint_step = previous_step + best_update
    with ocp.CheckpointManager(str(checkpoint), options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True, enable_async_checkpointing=False)) as manager:
        manager.save(checkpoint_step, args=ocp.args.StandardSave(nnx.state(model).to_pure_dict()))
    summary = {'previous_checkpoint': str(args.previous.resolve()), 'previous_step': previous_step, 'checkpoint': str(checkpoint), 'checkpoint_step': checkpoint_step, 'actual_updates': actual_updates, 'selected_best_update': best_update, 'stop_reason': stop_reason, 'max_updates': args.max_updates, 'early_stop': {'eval_every': args.eval_every, 'patience': args.patience, 'min_updates': args.min_updates, 'relative_improvement': args.relative_improvement}, 'initial_heldout_all_loss': initial_all, 'initial_heldout_latest_loss': initial_latest, 'heldout_all_loss': final_all, 'heldout_latest_loss': final_latest, 'cumulative_train_windows': len(data.train), 'cumulative_heldout_windows': len(data.heldout), 'latest_heldout_windows': len(data.latest), 'frozen_parameter_changes': frozen_changes, 'dynamics_parameter_change_count': len(dynamics_changes), 'optimizer': 'fresh optax.adam per stage, released fine-tune schedule', 'duration_seconds': time.monotonic() - started}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('TRAIN_SUMMARY', json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
