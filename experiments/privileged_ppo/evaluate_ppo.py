"""Sequential single-env PPO evaluation with complete step/segment evidence."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import argparse
from pathlib import Path
from datetime import datetime, timezone
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=Path)
parser.add_argument('--seeds', required=True)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--mass', type=float, default=8.0)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--stochastic', action='store_true')
args = parser.parse_args()
assert args.mass in [0.0, 8.0]
args.output.mkdir(parents=True, exist_ok=False)
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
app = launcher.app
import csv
import hashlib
import json
import random
import time
import traceback
import numpy as np
import torch
from isaacsim.core.utils.stage import create_new_stage
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.modules import ActorCritic
from target_env import configure, runtime_readback, write, digest
env = None
try:
    policy = ActorCritic(433, 433, 12, actor_hidden_dims=[512] * 3, critic_hidden_dims=[512] * 3, activation='elu', init_noise_std=1.0)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    policy.load_state_dict(ckpt['model_state_dict'])
    assert 'obs_norm_state_dict' not in ckpt and 'critic_obs_norm_state_dict' not in ckpt
    policy = policy.to('cuda:0').eval()
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    summaries, segments, readbacks = ([], [], [])
    for index, seed in enumerate(map(int, args.seeds.split(','))):
        if index:
            create_new_stage()
            app.update()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cfg = configure(seed, 1, args.mass)
        write(args.output / f'config_{seed}.json', cfg.to_dict())
        env = ManagerBasedRLEnv(cfg)
        obs, _ = env.reset()
        initial = runtime_readback(env, args.mass, f'seed_{seed}_initial')
        records, episode = ([], 0)
        segment_start = 0
        segment_return = 0.0
        seed_start = time.monotonic()
        for step in range(args.steps):
            with torch.inference_mode():
                privileged = obs['policy']
                assert privileged.shape == (1, 433)
                action = policy.act(privileged) if args.stochastic else policy.act_inference(privileged)
                value = policy.evaluate(privileged).item()
                vx = env.scene['robot'].data.root_lin_vel_b[0, 0].item()
                command = env.command_manager.get_command('base_velocity')[0].detach().cpu().numpy().copy()
                prop = np.concatenate([obs['obs'][n][0].detach().cpu().numpy() for n in ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 'joint_pos', 'joint_vel', 'cos_sin_phase']])
                assert np.isfinite(prop).all() and torch.isfinite(action).all() and np.isfinite(value)
                pre_obs = privileged[0].detach().cpu().numpy().copy()
                action_numpy = action[0].detach().cpu().numpy().copy()
                obs, reward, terminated, timeout, _ = env.step(action)
            r = float(reward[0])
            term, tout = (bool(terminated[0]), bool(timeout[0]))
            assert np.isfinite(r)
            records.append({'reward': r, 'vx': vx, 'command': command, 'action': action_numpy, 'privileged_obs': pre_obs, 'value': value, 'terminated': term, 'timeout': tout, 'episode': episode, 'proprio': prop, 'computed_torque': env.scene['robot'].data.computed_torque[0].detach().cpu().numpy().copy(), 'applied_torque': env.scene['robot'].data.applied_torque[0].detach().cpu().numpy().copy(), 'joint_target': env.scene['robot'].data.joint_pos_target[0].detach().cpu().numpy().copy(), 'reward_terms': (env.reward_manager._step_reward[0] * env.step_dt).detach().cpu().numpy().copy()})
            segment_return += r
            if term or tout:
                reason = 'terminated' if term else 'timeout'
                segments.append({'eval_seed': seed, 'segment_id': episode, 'start_step': segment_start, 'length': step + 1 - segment_start, 'Return': segment_return, 'end_reason': reason, 'ET_event': int(term), 'checkpoint_sha256': checkpoint_sha})
                readbacks.append(runtime_readback(env, args.mass, f'seed_{seed}_reset_{episode}', include_obs=False))
                episode += 1
                segment_start = step + 1
                segment_return = 0.0
        if segment_start < args.steps:
            segments.append({'eval_seed': seed, 'segment_id': episode, 'start_step': segment_start, 'length': args.steps - segment_start, 'Return': segment_return, 'end_reason': 'budget_censored', 'ET_event': 0, 'checkpoint_sha256': checkpoint_sha})
        step_path = args.output / f'raw_seed_{seed}.npz'
        np.savez_compressed(step_path, reward=np.array([r['reward'] for r in records], dtype=np.float32), actual_vx=np.array([r['vx'] for r in records]), command=np.array([r['command'] for r in records]), action=np.array([r['action'] for r in records]), privileged_obs=np.array([r['privileged_obs'] for r in records]), critic_value=np.array([r['value'] for r in records]), terminated=np.array([r['terminated'] for r in records]), timeout=np.array([r['timeout'] for r in records]), episode=np.array([r['episode'] for r in records]), proprio=np.array([r['proprio'] for r in records]), reward_terms=np.array([r['reward_terms'] for r in records]), computed_torque=np.array([r['computed_torque'] for r in records]), applied_torque=np.array([r['applied_torque'] for r in records]), joint_target=np.array([r['joint_target'] for r in records]), reward_term_names=np.array(env.reward_manager.active_terms), global_step=np.arange(args.steps))
        reward_values = np.array([r['reward'] for r in records], dtype=np.float64)
        np.testing.assert_allclose(np.array([r['reward_terms'] for r in records]).sum(axis=1), reward_values, atol=2e-08, rtol=2e-06)
        et = sum((r['terminated'] for r in records))
        summaries.append({'eval_seed': seed, 'Return': float(reward_values.sum()), 'ET_seed': int(et > 0), 'ET_events': et, 'timeout_count': sum((r['timeout'] and (not r['terminated']) for r in records)), 'executed_steps': args.steps, 'segments': len([s for s in segments if s['eval_seed'] == seed]), 'vx_RMSE': float(np.sqrt(np.mean((np.array([r['vx'] for r in records]) - 0.5) ** 2))), 'mean_action_norm': float(np.linalg.norm(np.array([r['action'] for r in records]), axis=1).mean()), 'wall_seconds': time.monotonic() - seed_start, 'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_sha, 'raw_path': str(step_path.resolve()), 'initial_state_sha256': initial['initial_root_joint_state_sha256'], 'terrain_sha256': initial['terrain_mesh_sha256'], 'reward_signature': initial['reward_signature'], 'termination_signature': initial['termination_signature'], 'action_signature': initial['action_signature'], 'mass': args.mass, 'policy_mode': 'stochastic' if args.stochastic else 'deterministic_mean', 'startup_zero_action_steps': 0})
        readbacks.append(initial)
        env.close()
        env = None
        for filename, rows in [('EVALUATION_RAW.csv', summaries), ('EPISODE_SEGMENTS.csv', segments)]:
            with (args.output / filename).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        write(args.output / 'RUNTIME_READBACK.json', readbacks)
        print('EVAL_SEED_COMPLETE', seed, summaries[-1]['Return'], 'ET', et, flush=True)
    write(args.output / 'result.json', {'status': 'PASS', 'n_seeds': len(summaries), 'steps_per_seed': args.steps, 'checkpoint_sha256': checkpoint_sha, 'started_argv': sys.argv, 'completed_utc': datetime.now(timezone.utc).isoformat()})
except BaseException:
    (args.output / 'failure.txt').write_text(traceback.format_exc())
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
