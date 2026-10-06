"""Continuous source PPO, paused for separate held-out single-env validation.

No PPO hyperparameters are altered. Validation subprocesses do not step or
reset the persistent training environment and have independent process RNGs.
"""
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
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--resume', type=Path)
args = parser.parse_args()
assert args.seed in [62001, 62002, 62003]
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
app = launcher.app
import csv
import hashlib
import json
import random
import shutil
import subprocess
import time
import traceback
import numpy as np
import torch
from scipy.stats import t
from torch.utils.tensorboard import SummaryWriter
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from simdist.rl.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
from target_env import ROOT, REPO, configure, runtime_readback, write
from recovery_support import select, offload_storage, restore_storage
OUT = ROOT / '02_training' / f'seed_{args.seed}'
OUT.mkdir(parents=True, exist_ok=args.resume is not None)
(OUT / 'checkpoints').mkdir(exist_ok=args.resume is not None)
env = None
completed = 0

class AuditVecEnvWrapper(RslRlVecEnvWrapper):

    @torch.inference_mode()
    def get_observations(self):
        return super().get_observations()

class AuditRunner(OnPolicyRunner):

    def log(self, locs, **kwargs):
        self.tot_timesteps += 4096 * 24
        self.tot_time += locs['collection_time'] + locs['learn_time']
        storage = self.alg.storage
        x = storage.observations.flatten(0, 1)
        old_mu = storage.mu.flatten(0, 1)
        old_sigma = storage.sigma.flatten(0, 1)
        kl_sum = 0.0
        with torch.inference_mode():
            sigma = self.alg.actor_critic.std
            assert torch.isfinite(sigma).all() and (sigma > 0).all()
            for start in range(0, len(x), 4096):
                mu = self.alg.actor_critic.act_inference(x[start:start + 4096])
                a = old_mu[start:start + 4096]
                b = old_sigma[start:start + 4096]
                kl = torch.sum(torch.log(sigma / b) + (b.square() + (a - mu).square()) / (2 * sigma.square()) - 0.5, dim=-1)
                kl_sum += float(kl.sum())
        metrics = {'train_seed': args.seed, 'target_iteration': locs['it'] + 1, 'target_env_steps': self.tot_timesteps, 'target_optimizer_updates': (locs['it'] + 1) * 20, 'value_loss': float(locs['mean_value_loss']), 'surrogate_loss': float(locs['mean_surrogate_loss']), 'entropy': float(locs['mean_entropy']), 'KL_full_update': kl_sum / len(x), 'learning_rate': self.alg.learning_rate, 'mean_action_std': float(sigma.mean()), 'min_action_std': float(sigma.min()), 'collection_seconds': locs['collection_time'], 'update_seconds': locs['learn_time'], 'train_episode_return': float(np.mean(locs['rewbuffer'])) if len(locs['rewbuffer']) else None, 'train_episode_length': float(np.mean(locs['lenbuffer'])) if len(locs['lenbuffer']) else None}
        assert all((np.isfinite(v) for v in metrics.values() if isinstance(v, (float, int))))
        curve = OUT / 'TRAIN_ITERATIONS.csv'
        with curve.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(metrics))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(metrics)
        for k in ['value_loss', 'surrogate_loss', 'entropy', 'KL_full_update', 'learning_rate', 'mean_action_std', 'train_episode_return']:
            if metrics[k] is not None:
                self.writer.add_scalar(k, metrics[k], metrics['target_env_steps'])
        if metrics['target_iteration'] % 25 == 0:
            print('TARGET_TRAIN_PROGRESS', json.dumps(metrics), flush=True)
            write(OUT / 'STATUS.json', {'phase': 'TRAINING', 'completed_target_iterations': metrics['target_iteration'], 'target_env_steps': metrics['target_env_steps'], 'updated_utc': datetime.now(timezone.utc).isoformat()})

def checkpoint(runner, iteration):
    path = OUT / 'checkpoints' / f'target_{iteration:06d}.pt'
    assert not path.exists()
    robot = env.scene['robot']
    checkpoint_state = {'model_state_dict': runner.alg.actor_critic.state_dict(), 'optimizer_state_dict': runner.alg.optimizer.state_dict(), 'iter': iteration, 'infos': None, 'target_iteration': iteration, 'target_env_steps': iteration * 4096 * 24, 'target_optimizer_updates': iteration * 20, 'source_iteration': 4999, 'source_checkpoint': str(checkpoint_root() / 'rl/go2_2026-02-11/model_4999.pt'), 'train_seed': args.seed, 'obs_normalizer_contract': 'Identity', 'critic_normalizer_contract': 'Identity', 'rng_state': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()}, 'environment_state': {'root_state_w': robot.data.root_state_w.clone(), 'joint_pos': robot.data.joint_pos.clone(), 'joint_vel': robot.data.joint_vel.clone(), 'joint_friction': robot.data.joint_friction.clone(), 'episode_length_buf': env.episode_length_buf.clone(), 'common_step_counter': env.common_step_counter, 'sim_step_counter': env._sim_step_counter, 'actions': env.action_manager.action.clone(), 'previous_actions': env.action_manager.prev_action.clone()}, 'resume_limit': 'Internal PhysX contacts/sensor histories not serialized; no exact replay claim on process restart', 'saved_utc': datetime.now(timezone.utc).isoformat()}
    torch.save(checkpoint_state, path)
    return (path, hashlib.sha256(path.read_bytes()).hexdigest())

def validation(path, sha, iteration):
    base = ROOT / '03_validation' / f'seed_{args.seed}' / f'target_{iteration:06d}'
    output = base
    attempt = 0
    while (output.exists() or (output.parent / f'{output.name}.log').exists()) and (not (output / 'result.json').exists()):
        attempt += 1
        output = base.with_name(f'{base.name}_retry{attempt:02d}')
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-u', str(PORTFOLIO_ROOT / 'experiments/privileged_ppo/evaluate_ppo.py'), '--checkpoint', str(path), '--seeds', ','.join(map(str, range(65001, 65011))), '--output', str(output)]
    write(OUT / 'STATUS.json', {'phase': 'VALIDATION', 'completed_target_iterations': iteration, 'validation_directory': str(output), 'updated_utc': datetime.now(timezone.utc).isoformat()})
    if not (output / 'result.json').exists():
        before = torch.cuda.memory_allocated()
        devices = offload_storage(runner.alg.storage)
        torch.cuda.empty_cache()
        write(OUT / f'VALIDATION_MEMORY_{iteration:06d}_attempt{attempt:02d}.json', {'torch_allocated_before': before, 'torch_allocated_after': torch.cuda.memory_allocated(), 'temporarily_offloaded_fields': sorted(devices), 'free_total_bytes': torch.cuda.mem_get_info(), 'physics_and_RNG_unchanged': True})
        try:
            with (output.parent / f'{output.name}.log').open('x') as log:
                proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        finally:
            restore_storage(runner.alg.storage, devices)
    result = output / 'result.json'
    assert result.exists() and json.loads(result.read_text())['status'] == 'PASS', f'Validation failed; inspect {output}'
    rows = list(csv.DictReader((output / 'EVALUATION_RAW.csv').open()))
    assert len(rows) == 10 and {int(r['eval_seed']) for r in rows} == set(range(65001, 65011))
    assert all((r['checkpoint_sha256'] == sha for r in rows))
    returns = np.array([float(r['Return']) for r in rows])
    et = np.array([int(r['ET_seed']) for r in rows])
    rmse = np.array([float(r['vx_RMSE']) for r in rows])
    stats = {'target_iteration': iteration, 'target_env_steps': iteration * 4096 * 24, 'mean_Return': float(returns.mean()), 'SD_Return': float(returns.std(ddof=1)), 'ET_seed_rate': float(et.mean()), 'ET_events': sum((int(r['ET_events']) for r in rows)), 'mean_vx_RMSE': float(rmse.mean()), 'per_seed_Return': returns.tolist(), 'checkpoint': str(path), 'checkpoint_sha256': sha}
    curves = OUT / 'PPO_TRAINING_CURVES.csv'
    existing = [r for r in csv.DictReader(curves.open()) if int(r['target_iteration']) == iteration] if curves.exists() else []
    if existing:
        assert len(existing) <= 10 and len({int(r['eval_seed']) for r in existing}) == len(existing)
        assert {int(r['eval_seed']) for r in existing} <= set(range(65001, 65011))
        assert all((r['checkpoint_sha256'] == sha for r in existing))
    missing = [r for r in rows if int(r['eval_seed']) not in {int(e['eval_seed']) for e in existing}]
    if missing:
        with curves.open('a', newline='') as f:
            fields = ['train_seed', 'target_iteration', 'target_env_steps', 'target_optimizer_updates', *rows[0].keys()]
            w = csv.DictWriter(f, fieldnames=fields)
            if f.tell() == 0:
                w.writeheader()
            for r in missing:
                w.writerow({'train_seed': args.seed, 'target_iteration': iteration, 'target_env_steps': iteration * 4096 * 24, 'target_optimizer_updates': iteration * 20, **r})
    print('VALIDATION_COMPLETE', args.seed, json.dumps(stats), flush=True)
    return stats

def save_selection(history, best, candidate, stop, confirmation):
    write(OUT / 'VALIDATION_SUMMARY.json', history)
    write(OUT / 'CHECKPOINT_SELECTION.json', {'best_validation': best, 'plateau_candidate': candidate, 'confirmation': confirmation, 'final': history[-1] if stop else None})

def save_result(item, best, confirmation):
    write(OUT / 'result.json', {'status': 'OPERATIONAL_PLATEAU_CONFIRMED', 'train_seed': args.seed, 'target_iterations': completed, 'target_env_steps': completed * 4096 * 24, 'target_optimizer_updates': completed * 20, 'final': item, 'best_validation': best, 'confirmation': confirmation, 'caveat': 'practical validation stability, not proof of optimality or critic accuracy'})
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
try:
    cfg = configure(args.seed, 4096, 8.0)
    config_prefix = f"RESUME_{args.resume.stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_" if args.resume else ''
    write(OUT / f'{config_prefix}effective_env.json', cfg.to_dict())
    agent = UnitreeGo2RoughPPORunnerCfg()
    agent.seed = args.seed
    agent.logger = 'tensorboard'
    agent.device = 'cuda:0'
    write(OUT / f'{config_prefix}effective_agent.json', agent.to_dict())
    env = ManagerBasedRLEnv(cfg)
    wrapped = AuditVecEnvWrapper(env)
    runner = AuditRunner(wrapped, agent.to_dict(), log_dir=str(OUT / 'source_runner'), device='cuda:0')
    load_path = args.resume or checkpoint_root() / 'rl/go2_2026-02-11/model_4999.pt'
    if args.resume:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert not runner.empirical_normalization and (not runner.alg.rnd)
        runner.alg.actor_critic.load_state_dict(resumed['model_state_dict'])
        runner.alg.optimizer.load_state_dict(resumed['optimizer_state_dict'])
    else:
        runner.load(str(load_path), load_optimizer=True)
    runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]['lr']
    runner.current_learning_iteration = 0
    runner.tot_timesteps = 0
    runner.tot_time = 0
    if args.resume:
        assert args.resume.resolve().parent == OUT.resolve() / 'checkpoints'
        resumed = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert resumed['train_seed'] == args.seed
        completed = resumed['target_iteration']
        runner.current_learning_iteration = completed
        runner.tot_timesteps = completed * 4096 * 24
        random.setstate(resumed['rng_state']['python'])
        np.random.set_state(resumed['rng_state']['numpy'])
        torch.set_rng_state(resumed['rng_state']['torch_cpu'])
        torch.cuda.set_rng_state_all(resumed['rng_state']['torch_cuda'])
        write(OUT / f'RESUME_{completed:06d}.json', {'resumed_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint': str(args.resume), 'sha256': hashlib.sha256(args.resume.read_bytes()).hexdigest(), 'restored': 'actor/critic/optimizer/Identity normalizer contract/Python+NumPy+Torch RNG/counters', 'environment': 'reconstructed fixed same-seed4096 envs; PhysX contact/sensor/command-history state is not exact-replayed', 'repair': 'initial observation read under inference mode; source PPO/hyperparameters/reward/physics unchanged'})
        write(OUT / f'RESUME_RUNTIME_{completed:06d}.json', runtime_readback(env, 8.0, 'resume_reconstructed'))
        write(OUT / 'STATUS.json', {'phase': 'RESUMING', 'completed_target_iterations': completed})
    else:
        write(OUT / 'INITIAL_RUNTIME_READBACK.json', runtime_readback(env, 8.0, 'training_initial'))
    observations, _ = wrapped.get_observations()
    assert observations.shape == (4096, 433)
    stop_after_pending_validation = False
    if args.resume:
        history = json.loads((OUT / 'VALIDATION_SUMMARY.json').read_text())
        selection = json.loads((OUT / 'CHECKPOINT_SELECTION.json').read_text())
        candidate = selection['plateau_candidate']
        best = selection['best_validation']
        assert history[-1]['target_iteration'] in [completed, completed - 250]
        if history[-1]['target_iteration'] < completed:
            history.append(validation(args.resume, hashlib.sha256(args.resume.read_bytes()).hexdigest(), completed))
            best, candidate, stop, confirmation = select(history, best, candidate)
            save_selection(history, best, candidate, stop, confirmation)
            if stop:
                save_result(history[-1], best, confirmation)
                stop_after_pending_validation = True
    else:
        history = []
        candidate = None
        best = None
        path, sha = checkpoint(runner, 0)
        history.append(validation(path, sha, 0))
        best = history[-1]
        write(OUT / 'CHECKPOINT_SELECTION.json', {'best_validation': best, 'plateau_candidate': None, 'final': None})
    first_target = args.resume is None
    initialize_episode_lengths = True
    while not stop_after_pending_validation:
        if shutil.disk_usage(ROOT).free < 4 * 1024 ** 3:
            raise RuntimeError('Disk free below4GiB; hardware storage decision required')
        start_utc = datetime.now(timezone.utc).isoformat()
        if first_target:
            write(OUT / 'TARGET_TRAIN_START.json', {'started_utc': start_utc, 'mode': 'A', 'source_iteration': 4999, 'target_iteration': 0})
        runner.learn(num_learning_iterations=250, init_at_random_ep_len=initialize_episode_lengths)
        initialize_episode_lengths = False
        first_target = False
        completed += 250
        runner.current_learning_iteration = completed
        path, sha = checkpoint(runner, completed)
        item = validation(path, sha, completed)
        history.append(item)
        best, candidate, stop, confirmation = select(history, best, candidate)
        save_selection(history, best, candidate, stop, confirmation)
        write(OUT / 'FINAL_RUNTIME_READBACK.json', runtime_readback(env, 8.0, f'target_iteration_{completed}', include_obs=False))
        if stop:
            save_result(item, best, confirmation)
            break
except BaseException:
    failure = OUT / ('failure.txt' if args.resume is None else f"failure_resume_{completed:06d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.txt")
    failure.write_text(traceback.format_exc())
    write(OUT / 'STATUS.json', {'phase': 'FAILED', 'last_completed_target_iterations': completed, 'failed_utc': datetime.now(timezone.utc).isoformat()})
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
