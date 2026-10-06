"""Evaluate every saved model on each cumulative held-out data split."""
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
import csv
import json
from pathlib import Path
import numpy as np
from flax import nnx
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from simdist.modeling import losses
from simdist.utils import model as model_utils
from curve_data import CurveData
ap = argparse.ArgumentParser()
ap.add_argument('--models', nargs='+', required=True, help='name=checkpoint_path')
ap.add_argument('--rollouts', nargs='+', required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()

def main():
    specs = [part.split('=', 1) for part in args.models]
    assert all((len(part) == 2 for part in specs))
    rows = []
    for model_name, checkpoint in specs:
        model, model_cfg, model_step = model_utils.load_model_from_ckpt(str(resolve_artifact_path(checkpoint).resolve()), 367004 if model_name == 'm0' else None)
        with initialize_config_dir(config_dir=str(simdist_root() / 'config'), version_base=None):
            cfg = OmegaConf.to_container(compose(config_name='finetune_model'), resolve=True)
        cfg['finetune'] = True
        cfg['model'] = model_cfg['model']
        cfg['system'] = model_cfg['system']
        cfg['scaler_params_struct'] = model_cfg['scaler_params_struct']
        loss_fn = losses.get_loss(cfg)

        @nnx.jit
        def eval_step(model, x, y):
            return loss_fn(model, x, y, True)[0]
        for count in range(1, len(args.rollouts) + 1):
            data = CurveData([str(resolve_artifact_path(p).resolve()) for p in args.rollouts[:count]], model_cfg)
            for split_name, refs in (('cumulative', data.heldout), ('latest', data.latest)):
                total = 0.0
                for offset in range(0, len(refs), 32):
                    selected = refs[offset:offset + 32]
                    batch = model_utils.dataset_batch_to_jax(data.batch(selected, augment=False))
                    total += float(eval_step(model, batch['model_in'], batch['labels'])) * len(selected)
                value = total / len(refs)
                assert np.isfinite(value)
                row = {'model': model_name, 'checkpoint': str(resolve_artifact_path(checkpoint).resolve()), 'model_step': model_step, 'data_stage': count, 'cumulative_control_steps': sum((x['control_steps'] for x in data.intervals)), 'split': split_name, 'windows': len(refs), 'latent_loss': value}
                rows.append(row)
                print('LOSS', json.dumps(row), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
if __name__ == '__main__':
    main()
