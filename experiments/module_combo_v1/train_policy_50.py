"""Fit only policy heads for exactly 50 updates with the oracle schedule."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.65')
import copy
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml
import flax.nnx as nnx
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from simdist.modeling import losses
from simdist.utils import model as model_utils
ROOT = artifact_dir('module_combo_v1')
REPO = PORTFOLIO_ROOT
from explicit_data import ExplicitSeedData
PARSER = argparse.ArgumentParser()
PARSER.add_argument('--previous', type=Path, required=True)
PARSER.add_argument('--train-rollouts', type=Path, nargs=5, required=True)
PARSER.add_argument('--heldout-rollout', type=Path, required=True)
PARSER.add_argument('--out', type=Path, required=True)
ARGS = PARSER.parse_args()

class ExpertSeedData(ExplicitSeedData):

    def batch(self, refs, augment=False):
        batch = super().batch(refs, augment=augment)
        batch['labels']['exp_pol_flags'][:] = True
        return batch

def flatten(tree, prefix=()):
    if isinstance(tree, dict):
        result = {}
        for key, value in tree.items():
            result.update(flatten(value, prefix + (str(key),)))
        return result
    return {'/'.join(prefix): np.asarray(tree)}

def main():
    started = time.monotonic()
    seed = 34501
    np.random.seed(seed)
    random.seed(seed)
    prior = ARGS.previous.resolve()
    out = ARGS.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    model, model_cfg, prior_step = model_utils.load_model_from_ckpt(str(prior), None)
    with initialize_config_dir(config_dir=str(simdist_root() / 'config'), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name='finetune_model'), resolve=True)
    cfg['finetune'] = True
    cfg['model'] = copy.deepcopy(model_cfg['model'])
    cfg['system'] = copy.deepcopy(model_cfg['system'])
    cfg['scaler_params_struct'] = copy.deepcopy(model_cfg['scaler_params_struct'])
    cfg['training']['seed'] = seed
    cfg['training']['training_data_ratio'] = 1.0
    assert cfg['heads'] == ['dynamics']
    cfg['heads'] = ['policy', 'policy_dec']
    cfg['loss']['weights'] = {'latent_dynamics': 0.0, 'reward': 0.0, 'value': 0.0, 'action': 1.0}
    assert cfg['training']['batch_size'] == 32
    train = [p.resolve() for p in ARGS.train_rollouts]
    heldout = ARGS.heldout_rollout.resolve()
    data = ExpertSeedData(train, [heldout], heldout, model_cfg)
    assert len(data.train) > 0 and len(data.heldout) > 0
    filt = model_utils.create_param_filter(cfg['heads'])
    frozen_filter = nnx.All(nnx.Param, nnx.Not(filt))
    frozen_before = flatten(nnx.state(model, frozen_filter).to_pure_dict())
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

    def evaluate():
        total = 0.0
        for start in range(0, len(data.heldout), 32):
            selected = data.heldout[start:start + 32]
            batch = model_utils.dataset_batch_to_jax(data.batch(selected, augment=False))
            total += float(eval_step(model, batch['model_in'], batch['labels'])) * len(selected)
        return total / len(data.heldout)
    curve = [{'update': 0, 'train_loss_recent': '', 'heldout_action_loss': evaluate(), 'learning_rate': 0.0}]
    order = np.random.permutation(len(data.train))
    cursor = 0
    train_values = []
    saved = []
    for update in range(1, 51):
        if cursor + 32 > len(order):
            order = np.random.permutation(len(data.train))
            cursor = 0
        indices = order[cursor:cursor + 32]
        cursor += 32
        batch = model_utils.dataset_batch_to_jax(data.batch([data.train[int(i)] for i in indices], augment=True))
        value = float(train_step(model, optimizer, batch['model_in'], batch['labels']))
        assert np.isfinite(value)
        train_values.append(value)
        if update == 50:
            heldout_loss = evaluate()
            row = {'update': update, 'train_loss_recent': float(np.mean(train_values)), 'heldout_action_loss': heldout_loss, 'learning_rate': float(lr(update - 1))}
            curve.append(row)
            print('PROGRESS', json.dumps(row), flush=True)
            train_values.clear()
        if update == 50:
            checkpoint = out / 'checkpoint'
            checkpoint.mkdir(parents=True, exist_ok=False)
            with (checkpoint / 'model_config.yaml').open('w') as stream:
                yaml.safe_dump(cfg, stream, sort_keys=False)
            step = prior_step + update
            with ocp.CheckpointManager(str(checkpoint), options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True, enable_async_checkpointing=False)) as manager:
                manager.save(step, args=ocp.args.StandardSave(nnx.state(model).to_pure_dict()))
            frozen_now = flatten(nnx.state(model, frozen_filter).to_pure_dict())
            assert all((np.array_equal(frozen_before[k], frozen_now[k]) for k in frozen_before))
            saved.append({'update': update, 'checkpoint_step': step, 'checkpoint': str(checkpoint.resolve()), 'heldout_action_loss': heldout_loss, 'learning_rate': float(lr(update - 1))})
    with (out / 'training_curve.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    (out / 'summary.json').write_text(json.dumps({'prior': str(prior), 'train_rollouts': [str(p) for p in train], 'heldout_rollout': str(heldout), 'saved': saved, 'train_windows': len(data.train), 'heldout_windows': len(data.heldout), 'training_seed': seed, 'updates': 50, 'frozen_parameter_changes': 0, 'duration_seconds': time.monotonic() - started}, indent=2) + '\n')
    print('SUMMARY', json.dumps(saved), flush=True)
if __name__ == '__main__':
    main()
