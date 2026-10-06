"""On-policy target collection using the v2 joint reset and expert warm-up."""
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
import logging
import sys
import time
from pathlib import Path
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', type=Path, required=True)
ap.add_argument('--step', type=int)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--gap', choices=('source', 'mass8'), required=True)
ap.add_argument('--terrain', choices=('plane', 'boxes'), default='boxes')
ap.add_argument('--friction', type=float, choices=(0.8, 0.15, 0.1), default=0.8)
ap.add_argument('--forward-vel', type=float, choices=(0.5, 1.0), default=1.0)
ap.add_argument('--steps', type=int, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
app = AppLauncher(headless=True).app
import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from isaaclab.envs import ManagerBasedRLEnv
from simdist.utils import model as model_utils, config as config_utils
from simdist.control.mppi import MppiController
from go2_probe_common import ROOT, EXPERT
from condition_config import make_condition, verify_condition
logging.getLogger('jax').setLevel(logging.WARNING)

def main():
    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    with initialize_config_dir(config_dir=str(simdist_root() / 'config'), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name='simulate_go2'), resolve=True)
    cfg['control']['seed'] = args.seed
    model, model_cfg, step = model_utils.load_model_from_ckpt(str(args.checkpoint.resolve()), args.step)
    names = config_utils.proprio_obs_names_from_sys_config(model_cfg['system'])
    controller = MppiController(model, model_cfg, cfg['control'])
    expert = torch.jit.load(str(EXPERT), map_location='cuda:0').eval()
    ec = make_condition(args.seed, args.gap, args.terrain, args.friction, args.forward_vel)
    ec.episode_length_s = 20.5
    env = ManagerBasedRLEnv(ec)
    try:
        obs, _ = env.reset()
        robot = env.scene['robot']
        physical = verify_condition(env, args.gap, args.terrain, args.friction)
        zero = np.zeros(12, np.float32)
        previous = zero.copy()

        def repair_joint_state():
            nonlocal obs
            robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
            env.sim.forward()
            env.scene.update(env.physics_dt)
            obs = env.observation_manager.compute()
            assert float(torch.max(torch.abs(robot.data.joint_pos - robot.data.default_joint_pos))) < 1e-05
            assert float(torch.max(torch.abs(robot.data.joint_vel))) < 1e-05

        def model_input(ob, prev):
            proprio = np.concatenate([ob['obs'][n][0].detach().cpu().numpy() for n in names]).astype(np.float32)
            extero = ob['obs']['height_scan'][0].detach().cpu().numpy().astype(np.float32)
            assert proprio.shape == (35,) and extero.shape == (315,)
            return {'proprio_obs': proprio, 'extero_obs': extero, 'prev_action': prev}
        repair_joint_state()
        command = env.command_manager.get_command('base_velocity')[0].detach().cpu().numpy().copy()
        controller.initialize(model_input(obs, zero), command)
        controller.reset(model_input(obs, zero), command)
        warmup_steps_total = 0
        warmup_terminations = 0
        warmup_attempts = 0

        def warmup():
            nonlocal obs, previous, warmup_steps_total, warmup_terminations, warmup_attempts
            while True:
                warmup_attempts += 1
                if warmup_attempts > args.steps // 50 + 100:
                    raise RuntimeError('Too many warm-up resets')
                previous = zero.copy()
                command = env.command_manager.get_command('base_velocity')[0].detach().cpu().numpy().copy()
                controller.reset(model_input(obs, zero), command)
                completed = True
                for _ in range(25):
                    command = env.command_manager.get_command('base_velocity')[0].detach().cpu().numpy().copy()
                    controller.set_fut_cmd(command)
                    controller.update(model_input(obs, previous))
                    with torch.inference_mode():
                        action = expert(obs['policy']).squeeze(0).detach().cpu().numpy().astype(np.float32)
                    obs, _, terminated, truncated, _ = env.step(torch.from_numpy(action[None]).to(env.device))
                    warmup_steps_total += 1
                    previous = action
                    if bool(terminated[0]) or bool(truncated[0]):
                        warmup_terminations += 1
                        repair_joint_state()
                        completed = False
                        break
                if completed:
                    return
        warmup()
        episodes = []
        current = {k: [] for k in ('proprio_obs', 'extero_obs', 'actions', 'commands', 'reward')}
        total = 0
        falls = 0
        with h5py.File(args.out / 'rollout.h5', 'x') as store:

            def finish(reason):
                nonlocal current
                if not current['actions']:
                    return
                group = store.create_group(f'episode_{len(episodes):04d}')
                for key, values in current.items():
                    group.create_dataset(key, data=np.asarray(values, dtype=np.float32), compression='gzip', compression_opts=1)
                group.attrs['end_reason'] = reason
                group.attrs['length'] = len(current['actions'])
                episodes.append({'index': len(episodes), 'length': len(current['actions']), 'end_reason': reason, 'return': float(np.sum(current['reward'], dtype=np.float64))})
                current = {k: [] for k in current}
                store.flush()
            while total < args.steps:
                command = env.command_manager.get_command('base_velocity')[0].detach().cpu().numpy().copy()
                controller.set_fut_cmd(command)
                x = model_input(obs, previous)
                controller.update(x)
                action = np.asarray(controller.run_control()['actions'][0], dtype=np.float32)
                assert action.shape == (12,) and np.isfinite(action).all()
                current['proprio_obs'].append(x['proprio_obs'].copy())
                current['extero_obs'].append(x['extero_obs'].copy())
                current['actions'].append(action.copy())
                current['commands'].append(command.copy())
                obs, reward, terminated, truncated, _ = env.step(torch.from_numpy(action[None]).to(env.device))
                current['reward'].append(float(reward[0]))
                previous = action
                total += 1
                if bool(terminated[0]) or bool(truncated[0]):
                    reason = 'terminated' if bool(terminated[0]) else 'timeout'
                    if reason == 'terminated':
                        falls += 1
                    finish(reason)
                    repair_joint_state()
                    if total < args.steps:
                        warmup()
                if total % 100 == 0:
                    print(f'COLLECTION_PROGRESS {total}/{args.steps} falls={falls}', flush=True)
            finish('budget_censored')
        assert sum((ep['length'] for ep in episodes)) == args.steps
        summary = {'seed': args.seed, 'gap': args.gap, 'checkpoint': str(args.checkpoint.resolve()), 'terrain': args.terrain, 'foot_friction': args.friction, 'forward_target_mps': args.forward_vel, 'checkpoint_step': step, 'requested_control_steps': args.steps, 'completed_control_steps': total, 'warmup_steps_excluded': warmup_steps_total, 'warmup_attempts': warmup_attempts, 'warmup_terminations': warmup_terminations, 'falls': falls, 'episodes': episodes, 'physical_readback': physical, 'duration_seconds': time.monotonic() - start}
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('COLLECTION_SUMMARY', json.dumps({k: v for k, v in summary.items() if k != 'episodes'}), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.close()
        app.close()
if __name__ == '__main__':
    main()
