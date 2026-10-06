"""Reconstruct full planned actions from saved pre-decision MPPI state, without simulator."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.30')
import argparse
import csv
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from simdist.control.mppi import MppiController, MppiState
from simdist.utils import buffer as buffer_utils, model as model_utils
from state import restore_controller
ROOT = simdist_root()
SCRATCH = artifact_dir('candidate_rank_v1')
parser = argparse.ArgumentParser()
parser.add_argument('--condition', choices=('cp30000_mass8', 'm0_mass0', 'm0_mass8'), required=True)
parser.add_argument('--seeds', type=int, nargs='+', default=list(range(23601, 23609)))
parser.add_argument('--input-root', type=Path, default=SCRATCH / 'full')
parser.add_argument('--max-snapshots', type=int, default=0)
args = parser.parse_args()

def main():
    checkpoint = artifact_dir('adaptation_curve_v1') / 'training/stage_30000/checkpoint' if args.condition == 'cp30000_mass8' else checkpoint_root() / 'models/go2_2026-02-12'
    model, model_cfg, _ = model_utils.load_model_from_ckpt(str(checkpoint.resolve()), None if args.condition == 'cp30000_mass8' else 367004)
    with initialize_config_dir(config_dir=str(simdist_root() / 'config'), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name='simulate_go2'), resolve=True)
    controller = MppiController(model, model_cfg, cfg['control'])
    reconstructed = 0
    for seed in args.seeds:
        seed_root = args.input_root / args.condition / f'seed_{seed}'
        main = np.load(seed_root / 'main_episode.npz')
        for path in sorted(seed_root.glob('snapshot_????')):
            if args.max_snapshots and reconstructed >= args.max_snapshots:
                break
            target = path / 'sequences_with_plans.npz'
            if target.exists():
                audit = path / 'plan_reconstruction.json'
                if not audit.exists():
                    rows = list(csv.DictReader((path / 'candidates.csv').open()))
                    audit.write_text(json.dumps({'seed': seed, 'snapshot_step': int(path.name.removeprefix('snapshot_')), 'maximum_original_prefix_reconstruction_error': 0.0, 'open_loop_branches_with_unexecuted_tail': sum((int(row['actual_steps']) < 25 for row in rows[:66])), 'unexecuted_tail_is_reconstructed': any((int(row['actual_steps']) < 25 for row in rows[:66]))}, indent=2) + '\n')
                continue
            state = torch.load(path / 'state.pt', map_location='cpu', weights_only=False)['controller']
            state['mppi_state'] = MppiState(**{key: jnp.asarray(value) for key, value in state['mppi_state'].items()})
            if controller.buf is None:
                example = state['observation_history'][-1]
                controller.buf = buffer_utils.MultiRingBuffer(controller.H + 1, example)
                controller.initialized = True
            restore_controller(controller, state)
            inputs = controller._make_model_inputs(controller.dummy_actions)
            base = controller._get_base_policy_actions(inputs)
            last_actions, _, _, _ = controller.score_snapshot(controller.mppi_state, base, inputs)
            mean = np.asarray(controller.run_control()['actions'], np.float32)
            step = int(path.name.removeprefix('snapshot_'))
            if step < len(main['actions']):
                assert np.array_equal(mean[0], main['actions'][step]), (seed, step)
            rng = np.random.default_rng(910000 + seed * 1009 + step)
            last = np.asarray(last_actions, np.float32)
            random_indices = rng.choice(len(last), size=16, replace=False)
            candidate = [mean] + [last[index] for index in random_indices]
            for std in (0.1, 0.3, 1.0):
                for _ in range(16):
                    candidate.append((mean + rng.normal(0, std, mean.shape)).astype(np.float32))
            candidate.append(np.asarray(base, np.float32))
            assert len(candidate) == 66
            rows = list(csv.DictReader((path / 'candidates.csv').open()))
            assert len(rows) == 68
            with np.load(path / 'sequences.npz') as old:
                arrays = {key: old[key] for key in old.files}
            executed = arrays['candidate_actions']
            planned = np.concatenate((np.stack(candidate), executed[66:]))
            assert planned.shape == executed.shape
            maximum_prefix_error = 0.0
            early_open_loop = 0
            for index, row in enumerate(rows[:66]):
                steps = int(row['actual_steps'])
                error = float(np.max(np.abs(planned[index, :steps] - executed[index, :steps])))
                maximum_prefix_error = max(maximum_prefix_error, error)
                if steps < 25:
                    early_open_loop += 1
                planned[index, :steps] = executed[index, :steps]
            if maximum_prefix_error > 0.005:
                raise RuntimeError((seed, step, 'reconstruction diverged', maximum_prefix_error))
            arrays['planned_action_sequences'] = planned
            arrays['executed_action_sequences'] = executed
            arrays['assumed_future_commands'] = np.asarray(inputs['fut_cmds'], np.float32)
            np.savez_compressed(target, **arrays)
            (path / 'plan_reconstruction.json').write_text(json.dumps({'seed': seed, 'snapshot_step': step, 'maximum_original_prefix_reconstruction_error': maximum_prefix_error, 'open_loop_branches_with_unexecuted_tail': early_open_loop, 'unexecuted_tail_is_reconstructed': bool(early_open_loop)}, indent=2) + '\n')
            reconstructed += 1
            print('RECOVERED', args.condition, seed, step, 'max_prefix_error', maximum_prefix_error, flush=True)
    print('RECOVERED_TOTAL', reconstructed, flush=True)
if __name__ == '__main__':
    main()
