"""Collect v2 target expert-policy rollouts for a policy-head oracle fit."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.10')
import argparse
import json
import sys
import time
from pathlib import Path
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--gap', choices=('friction015', 'gain070'), required=True)
ap.add_argument('--steps', type=int, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
app = AppLauncher(headless=True).app
import h5py
import numpy as np
import torch
import yaml
from isaaclab.envs import ManagerBasedRLEnv
from simdist.utils import config as config_utils
from go2_probe_common import ROOT, MODEL, EXPERT
from gap_config import make_gap, verify_gap

def main():
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    model_cfg = yaml.safe_load((MODEL / 'model_config.yaml').read_text())
    names = config_utils.proprio_obs_names_from_sys_config(model_cfg['system'])
    expert = torch.jit.load(str(EXPERT), map_location='cuda:0').eval()
    ec = make_gap(args.seed, args.gap)
    ec.episode_length_s = 20.5
    env = ManagerBasedRLEnv(ec)
    try:
        obs, _ = env.reset()
        robot = env.scene['robot']
        physical = verify_gap(env, args.gap)

        def repair_joint_state():
            nonlocal obs
            robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
            env.sim.forward()
            env.scene.update(env.physics_dt)
            obs = env.observation_manager.compute()
            assert float(torch.max(torch.abs(robot.data.joint_pos - robot.data.default_joint_pos))) < 1e-05
            assert float(torch.max(torch.abs(robot.data.joint_vel))) < 1e-05

        def model_observation(ob):
            proprio = np.concatenate([ob['obs'][name][0].detach().cpu().numpy() for name in names]).astype(np.float32)
            extero = ob['obs']['height_scan'][0].detach().cpu().numpy().astype(np.float32)
            assert proprio.shape == (35,) and extero.shape == (315,)
            return (proprio, extero)

        def action_from_expert(ob):
            with torch.inference_mode():
                action = expert(ob['policy']).squeeze(0).detach().cpu().numpy().astype(np.float32)
            assert action.shape == (12,) and np.isfinite(action).all()
            return action
        repair_joint_state()
        warmup_steps_total = 0
        warmup_terminations = 0
        warmup_attempts = 0

        def warmup():
            nonlocal obs, warmup_steps_total, warmup_terminations, warmup_attempts
            while True:
                warmup_attempts += 1
                if warmup_attempts > args.steps // 50 + 100:
                    raise RuntimeError('Too many warm-up resets')
                completed = True
                for _ in range(25):
                    action = action_from_expert(obs)
                    obs, _, terminated, truncated, _ = env.step(torch.from_numpy(action[None]).to(env.device))
                    warmup_steps_total += 1
                    if bool(terminated[0]) or bool(truncated[0]):
                        warmup_terminations += 1
                        repair_joint_state()
                        completed = False
                        break
                if completed:
                    return
        warmup()
        episodes = []
        current = {key: [] for key in ('proprio_obs', 'extero_obs', 'actions', 'commands', 'reward')}
        total = falls = 0
        with h5py.File(args.out / 'rollout.h5', 'x') as store:

            def finish(reason):
                nonlocal current
                if not current['actions']:
                    return
                group = store.create_group(f'episode_{len(episodes):04d}')
                for key, values in current.items():
                    group.create_dataset(key, data=np.asarray(values, np.float32), compression='gzip', compression_opts=1)
                group.attrs['end_reason'] = reason
                group.attrs['length'] = len(current['actions'])
                episodes.append({'index': len(episodes), 'length': len(current['actions']), 'end_reason': reason, 'return': float(np.sum(current['reward'], dtype=np.float64))})
                current = {key: [] for key in current}
                store.flush()
            while total < args.steps:
                command = env.command_manager.get_command('base_velocity')[0].detach().cpu().numpy().copy()
                proprio, extero = model_observation(obs)
                action = action_from_expert(obs)
                current['proprio_obs'].append(proprio.copy())
                current['extero_obs'].append(extero.copy())
                current['actions'].append(action.copy())
                current['commands'].append(command.copy())
                obs, reward, terminated, truncated, _ = env.step(torch.from_numpy(action[None]).to(env.device))
                current['reward'].append(float(reward[0]))
                total += 1
                if bool(terminated[0]) or bool(truncated[0]):
                    reason = 'terminated' if bool(terminated[0]) else 'timeout'
                    falls += int(reason == 'terminated')
                    finish(reason)
                    repair_joint_state()
                    if total < args.steps:
                        warmup()
                if total % 100 == 0:
                    print(f'COLLECTION_PROGRESS {total}/{args.steps} falls={falls}', flush=True)
            finish('budget_censored')
        assert sum((ep['length'] for ep in episodes)) == args.steps
        summary = {'seed': args.seed, 'gap': args.gap, 'completed_control_steps': total, 'falls': falls, 'warmup_steps_excluded': warmup_steps_total, 'warmup_attempts': warmup_attempts, 'warmup_terminations': warmup_terminations, 'episodes': episodes, 'physical_readback': physical, 'duration_seconds': time.monotonic() - started}
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('COLLECTION_SUMMARY', json.dumps({key: value for key, value in summary.items() if key != 'episodes'}), flush=True)
    finally:
        env.close()
        app.close()
if __name__ == '__main__':
    main()
