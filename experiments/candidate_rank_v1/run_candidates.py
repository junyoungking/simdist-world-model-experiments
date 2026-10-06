"""Scratch-only candidate-level same-state evaluation. No learning."""
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
import logging
import sys
import time
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--condition', choices=('cp30000_mass8', 'm0_mass0', 'm0_mass8'), required=True)
parser.add_argument('--seeds', type=int, nargs='+', required=True)
parser.add_argument('--out-root', type=Path, required=True)
parser.add_argument('--max-snapshots', type=int, default=0, help='Pilot only: process this many snapshots per episode')
parser.add_argument('--max-candidates', type=int, default=0, help='Pilot only: branch this many open-loop candidates')
args = parser.parse_args()
app = AppLauncher(headless=True).app
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from isaaclab.envs import ManagerBasedRLEnv
from simdist.control.mppi import MppiController
from simdist.utils import config as config_utils, model as model_utils
from go2_probe_common import ROOT, EXPERT, config, verify
from state import capture_env, restore_env, capture_controller, restore_controller
from model_heads import score_actions, observed_heads, state_value_from_observed_history
import omni.usd
logging.getLogger('jax').setLevel(logging.WARNING)
CRITIC = checkpoint_root() / 'rl/go2_2026-02-11/critics/critic_4999.pt'
SCRATCH = artifact_dir('candidate_rank_v1')
DISCOUNT = 0.99
T = 25

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def clone_obs(obs):
    return {key: {name: value.clone() for name, value in group.items()} if isinstance(group, dict) else group.clone() for key, group in obs.items()}

def from_obs(obs, previous, names):
    proprio = np.concatenate([obs['obs'][name][0].detach().cpu().numpy() for name in names])
    extero = obs['obs']['height_scan'][0].detach().cpu().numpy()
    assert proprio.shape == (35,) and extero.shape == (315,)
    return {'proprio_obs': proprio.astype(np.float32), 'extero_obs': extero.astype(np.float32), 'prev_action': previous.astype(np.float32)}

def get_cmd(env):
    return env.command_manager.get_command('base_velocity')[0].detach().cpu().numpy().astype(np.float32).copy()

def get_critic(critic, obs):
    with torch.inference_mode():
        return float(critic(obs['policy']).squeeze().item())

def get_expert_action(expert, obs):
    with torch.inference_mode():
        return expert(obs['policy']).squeeze(0).detach().cpu().numpy().astype(np.float32)

def export_snapshot(path, saved_env, saved_controller):
    copy = dict(saved_controller)
    state = copy['mppi_state']
    copy['mppi_state'] = {'mean': np.asarray(state.mean), 'std': np.asarray(state.std), 'key_data': np.asarray(state.key_data)}
    torch.save({'env': saved_env, 'controller': copy}, path)

def run_branch(env, controller, expert, critic, names, saved_env, saved_controller, kind, planned, max_steps, assumed_cmds):
    restore_env(env, saved_env)
    restore_controller(controller, saved_controller)
    obs = clone_obs(saved_controller['obs'])
    previous = saved_controller['previous'].copy()
    robot = env.scene['robot']
    initial_root_error = float(torch.max(torch.abs(robot.data.root_pos_w - saved_env['root_pos_audit'])).item())
    initial_joint_error = float(torch.max(torch.abs(robot.data.joint_pos - saved_env['joint_pos_audit'])).item())
    actions = []
    rewards = []
    commands_pre = []
    commands_post = []
    observed = [from_obs(obs, previous, names)]
    fell = False
    truncated = False
    reason = ''
    terminal_critic = np.nan
    for step in range(max_steps):
        command = get_cmd(env)
        commands_pre.append(command)
        if kind == 'expert_closed':
            action = get_expert_action(expert, obs)
        elif kind == 'base_closed':
            if step:
                controller.set_fut_cmd(command)
                controller.update(from_obs(obs, previous, names))
            inputs = controller._make_model_inputs(controller.dummy_actions)
            action = np.asarray(controller._get_base_policy_actions(inputs)[0], np.float32)
        else:
            action = np.array(planned[step], dtype=np.float32, copy=True)
        action = np.array(action, dtype=np.float32, copy=True)
        assert action.shape == (12,) and np.isfinite(action).all()
        obs, reward, terminated, timeout, _ = env.step(torch.from_numpy(action[None]).to(env.device))
        actions.append(action.copy())
        rewards.append(float(reward[0]))
        previous = action.copy()
        fell = bool(terminated[0])
        truncated = bool(timeout[0])
        reason = '|'.join((name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[0])))
        if fell or truncated:
            break
        commands_post.append(get_cmd(env))
        observed.append(from_obs(obs, previous, names))
        if step == 24:
            terminal_critic = get_critic(critic, obs)
    actual_commands = np.asarray(commands_pre[:25], np.float32)
    mismatch = float(np.mean(np.abs(actual_commands - assumed_cmds[:len(actual_commands)])))
    reward25 = float(sum((DISCOUNT ** i * reward for i, reward in enumerate(rewards[:25]))))
    reward200 = float(sum((DISCOUNT ** i * reward for i, reward in enumerate(rewards))))
    return {'kind': kind, 'steps': len(rewards), 'fell': fell, 'truncated': truncated, 'restore_root_error': initial_root_error, 'restore_joint_error': initial_joint_error, 'termination_terms': reason, 'fall_step': len(rewards) if fell else None, 'actions': actions, 'rewards': rewards, 'commands_pre': commands_pre, 'commands_post': commands_post, 'observed': observed, 'actual_command_mae': mismatch, 'reward25': reward25, 'reward200': reward200, 'terminal_critic': terminal_critic}

def prepare_observed_batch(branches):
    proprio = np.zeros((len(branches), 26, 35), np.float32)
    extero = np.zeros((len(branches), 26, 315), np.float32)
    actions = np.zeros((len(branches), 25, 12), np.float32)
    commands = np.zeros((len(branches), 26, 3), np.float32)
    for idx, branch in enumerate(branches):
        observed = branch['observed'][:26]
        action_list = branch['actions'][:25]
        command_pre = branch['commands_pre'][:25]
        command_post = branch['commands_post'][:25]
        for step in range(26):
            x = observed[min(step, len(observed) - 1)]
            proprio[idx, step] = x['proprio_obs']
            extero[idx, step] = x['extero_obs']
            commands[idx, step] = command_pre[0] if step == 0 else command_post[min(step - 1, len(command_post) - 1)] if command_post else command_pre[0]
        for step in range(25):
            actions[idx, step] = action_list[min(step, len(action_list) - 1)]
    return (proprio, extero, actions, commands)

def process_snapshot(env, controller, expert, critic, names, snapshot, out, seed, condition, limit_candidates):
    started = time.monotonic()
    saved_env, saved_controller = (snapshot['env'], snapshot['controller'])
    restore_env(env, saved_env)
    restore_controller(controller, saved_controller)
    command = get_cmd(env)
    controller.set_fut_cmd(command)
    inputs = controller._make_model_inputs(controller.dummy_actions)
    base_actions = controller._get_base_policy_actions(inputs)
    last_actions, _, _, _ = controller.score_snapshot(controller.mppi_state, base_actions, inputs)
    final_mean = np.asarray(controller.run_control()['actions'], np.float32)
    if snapshot['next_main_action'] is not None:
        assert np.array_equal(final_mean[0], snapshot['next_main_action']), 'Planner restore mismatch'
    rng = np.random.default_rng(910000 + seed * 1009 + snapshot['step'])
    last = np.asarray(last_actions, np.float32)
    random_indices = rng.choice(len(last), size=16, replace=False)
    candidates = [final_mean]
    kinds = ['mean_plan']
    indices = [-1]
    for index in random_indices:
        candidates.append(last[index])
        kinds.append('last_iteration')
        indices.append(int(index))
    for std in (0.1, 0.3, 1.0):
        for _ in range(16):
            candidates.append((final_mean + rng.normal(0, std, final_mean.shape)).astype(np.float32))
            kinds.append(f'noise_{std}')
            indices.append(-1)
    candidates.append(np.asarray(base_actions, np.float32))
    kinds.append('base_proposal')
    indices.append(-1)
    assert len(candidates) == 66
    if limit_candidates:
        candidates, kinds, indices = (candidates[:limit_candidates], kinds[:limit_candidates], indices[:limit_candidates])
    candidate_actions = np.stack(candidates)
    pred_rewards, pred_value, pred_reward_sum, pred_v_disc, pred_total = score_actions(controller.model, inputs, jnp.asarray(candidate_actions), controller.discounts, controller.final_discount)
    pred_rewards = np.asarray(pred_rewards, np.float32)
    pred_value = np.asarray(pred_value, np.float32)
    pred_reward_sum = np.asarray(pred_reward_sum, np.float32)
    pred_v_disc = np.asarray(pred_v_disc, np.float32)
    pred_total = np.asarray(pred_total, np.float32)
    assumed_cmds = np.asarray(inputs['fut_cmds'], np.float32)
    hobs = saved_controller['observation_history'][-25:]
    hcmd = np.asarray(saved_controller['command_history'][-25:], np.float32)
    hprop = np.stack([item['proprio_obs'] for item in hobs])[None]
    hextero = np.stack([item['extero_obs'] for item in hobs])[None]
    snapshot_value = float(np.asarray(state_value_from_observed_history(controller.model, jnp.asarray(hprop), jnp.asarray(hextero), jnp.asarray(hcmd[None])))[0])
    snapshot_critic = get_critic(critic, saved_controller['obs'])
    branches = []
    for index, (kind, actions) in enumerate(zip(kinds, candidates)):
        branch = run_branch(env, controller, expert, critic, names, saved_env, saved_controller, kind, actions, 25, assumed_cmds)
        branch['candidate_index'] = index
        branches.append(branch)
    for kind in ('expert_closed', 'base_closed'):
        branch = run_branch(env, controller, expert, critic, names, saved_env, saved_controller, kind, None, 200, assumed_cmds)
        branch['candidate_index'] = len(branches)
        branches.append(branch)
    mean_first_reward_error = abs(branches[0]['rewards'][0] - snapshot['next_main_reward']) if snapshot['next_main_reward'] is not None else np.nan
    observed = prepare_observed_batch(branches)
    latents, actual_head_rewards, actual_head_values = observed_heads(controller.model, *(jnp.asarray(item) for item in observed))
    latents = np.asarray(latents, np.float32)
    actual_head_rewards = np.asarray(actual_head_rewards, np.float32)
    actual_head_values = np.asarray(actual_head_values, np.float32)
    extra_actions = np.stack([observed[2][idx] for idx in range(len(candidates), len(branches))])
    e_rew, e_val, e_rew_sum, e_v_disc, e_total = score_actions(controller.model, inputs, jnp.asarray(extra_actions), controller.discounts, controller.final_discount)
    pred_rewards = np.concatenate((pred_rewards, np.asarray(e_rew, np.float32)))
    pred_value = np.concatenate((pred_value, np.asarray(e_val, np.float32)))
    pred_reward_sum = np.concatenate((pred_reward_sum, np.asarray(e_rew_sum, np.float32)))
    pred_v_disc = np.concatenate((pred_v_disc, np.asarray(e_v_disc, np.float32)))
    pred_total = np.concatenate((pred_total, np.asarray(e_total, np.float32)))
    episode_length = snapshot['episode_length']
    future_fall = snapshot['fall']
    remaining = episode_length - snapshot['step'] if future_fall else None
    rows = []
    actual_rewards = np.full((len(branches), 200), np.nan, np.float32)
    actual_commands = np.full((len(branches), 200, 3), np.nan, np.float32)
    for idx, branch in enumerate(branches):
        n = len(branch['rewards'])
        actual_rewards[idx, :n] = branch['rewards']
        actual_commands[idx, :n] = branch['commands_pre']
        full25 = n >= 25 and (not (branch['fell'] or branch['truncated'])) if n == 25 else n >= 25
        actual_head_r = float(np.sum(actual_head_rewards[idx] * DISCOUNT ** np.arange(25))) if full25 else np.nan
        actual_head_v = float(actual_head_values[idx, -1]) if full25 else np.nan
        pred_c = float(pred_reward_sum[idx])
        actual_total = branch['reward25'] + DISCOUNT ** 26 * branch['terminal_critic'] if full25 else branch['reward25']
        rows.append({'condition': condition, 'seed': seed, 'snapshot_step': snapshot['step'], 'episode_length': episode_length, 'episode_fell': int(future_fall), 'steps_to_main_fall': remaining if remaining is not None else '', 'near_main_fall_100': int(future_fall and remaining <= 100), 'candidate_index': idx, 'candidate_kind': kinds[idx] if idx < len(kinds) else branch['kind'], 'last_iteration_index': indices[idx] if idx < len(indices) else -1, 'pred_reward_discounted': pred_c, 'pred_terminal_value_raw': float(pred_value[idx]), 'pred_terminal_value_discounted': float(pred_v_disc[idx]), 'pred_total': float(pred_total[idx]), 'actual_steps': n, 'actual_fell': int(branch['fell']), 'actual_truncated': int(branch['truncated']), 'actual_fall_step': branch['fall_step'] or '', 'actual_termination_terms': branch['termination_terms'], 'actual_reward25_discounted': branch['reward25'], 'actual_reward200_discounted': branch['reward200'] if idx >= len(candidates) else '', 'actual_head_reward_discounted': actual_head_r if full25 else '', 'actual_head_terminal_value_raw': actual_head_v if full25 else '', 'expert_critic_step25': branch['terminal_critic'] if full25 else '', 'actual_total_with_critic': actual_total, 'snapshot_head_value_raw': snapshot_value, 'snapshot_expert_critic': snapshot_critic, 'command_mae': branch['actual_command_mae'], 'restore_root_error': branch['restore_root_error'], 'restore_joint_error': branch['restore_joint_error'], 'mean_first_reward_error': mean_first_reward_error, 'actual_25_complete': int(full25), 'expert_action_sequence_complete': int(idx < len(candidates) or n >= 25), 'pred_reward_sequence_file': str((out / 'sequences.npz').resolve())})
    write_csv(out / 'candidates.csv', rows)
    np.savez_compressed(out / 'sequences.npz', candidate_kinds=np.asarray([row['candidate_kind'] for row in rows]), planned_action_sequences=np.concatenate((candidate_actions, extra_actions)), executed_action_sequences=observed[2], assumed_future_commands=assumed_cmds, predicted_reward_sequence=pred_rewards, actual_env_reward_sequence=actual_rewards, actual_command_sequence=actual_commands, observed_latents=latents, actual_latent_reward_head_sequence=actual_head_rewards, actual_latent_value_head_sequence=actual_head_values, actual_head_valid=np.asarray([row['actual_25_complete'] for row in rows], bool))
    meta = {'condition': condition, 'seed': seed, 'step': snapshot['step'], 'episode_length': episode_length, 'main_fell': future_fall, 'steps_to_main_fall': remaining, 'candidate_count': len(candidates), 'branches': len(branches), 'snapshot_head_value_raw': snapshot_value, 'snapshot_expert_critic': snapshot_critic, 'mean_first_reward_error': mean_first_reward_error, 'max_restore_root_error': max((item['restore_root_error'] for item in branches)), 'max_restore_joint_error': max((item['restore_joint_error'] for item in branches)), 'branch_duration_seconds': time.monotonic() - started}
    (out / 'summary.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('SNAPSHOT_DONE', json.dumps(meta), flush=True)
    return rows

def main():
    started = time.monotonic()
    mass = 0.0 if args.condition == 'm0_mass0' else 8.0
    checkpoint = artifact_dir('adaptation_curve_v1') / 'training/stage_30000/checkpoint' if args.condition == 'cp30000_mass8' else checkpoint_root() / 'models/go2_2026-02-12'
    step = None if args.condition == 'cp30000_mass8' else 367004
    with initialize_config_dir(config_dir=str(simdist_root() / 'config'), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name='simulate_go2'), resolve=True)
    model, model_cfg, model_step = model_utils.load_model_from_ckpt(str(checkpoint.resolve()), step)
    names = config_utils.proprio_obs_names_from_sys_config(model_cfg['system'])
    controller = None
    expert = torch.jit.load(str(EXPERT), map_location='cuda:0').eval()
    critic = torch.jit.load(str(CRITIC), map_location='cuda:0').eval()
    for seed in args.seeds:
        seed_started = time.monotonic()
        out = args.out_root / args.condition / f'seed_{seed}'
        if (out / 'summary.json').exists():
            print('SEED_DONE', seed, flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        cfg['control']['seed'] = seed
        if controller is None:
            controller = MppiController(model, model_cfg, cfg['control'])
        else:
            controller.rngs = nnx.Rngs(seed)
            controller.rngs()
        ec = config(seed, mass, 1.0)
        ec.episode_length_s = 25.0
        env = ManagerBasedRLEnv(ec)
        try:
            obs, _ = env.reset()
            robot = env.scene['robot']
            robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
            env.sim.forward()
            env.scene.update(env.physics_dt)
            obs = env.observation_manager.compute()
            physical = verify(env, mass)
            zero = np.zeros(12, np.float32)
            previous = zero.copy()
            command = get_cmd(env)
            if not controller.initialized:
                controller.initialize(from_obs(obs, zero, names), command)
            controller.reset(from_obs(obs, zero, names), command)
            observation_history = [from_obs(obs, zero, names)]
            command_history = [command.copy()]
            warmup_terminations = 0
            for _ in range(25):
                command = get_cmd(env)
                controller.set_fut_cmd(command)
                controller.update(from_obs(obs, previous, names))
                action = get_expert_action(expert, obs)
                obs, _, terminated, timeout, _ = env.step(torch.from_numpy(action[None]).to(env.device))
                previous = action.copy()
                if bool(terminated[0]) or bool(timeout[0]):
                    warmup_terminations += 1
                    break
                observation_history.append(from_obs(obs, previous, names))
                command_history.append(get_cmd(env))
            assert warmup_terminations == 0, (seed, 'warmup failed')
            snapshots = []
            main_rewards = []
            main_actions = []
            main_fell = False
            reason = 'budget_censored'
            for t in range(1000):
                command = get_cmd(env)
                controller.set_fut_cmd(command)
                controller.update(from_obs(obs, previous, names))
                if t and t % 50 == 0:
                    saved_env = capture_env(env)
                    saved_controller = capture_controller(controller, previous, obs, observation_history[-25:], command_history[-25:])
                    snapshots.append({'step': t, 'env': saved_env, 'controller': saved_controller})
                action = np.asarray(controller.run_control()['actions'][0], np.float32)
                if t and t % 50 == 0:
                    snapshots[-1]['next_main_action'] = action.copy()
                obs, reward, terminated, timeout, _ = env.step(torch.from_numpy(action[None]).to(env.device))
                if t and t % 50 == 0:
                    snapshots[-1]['next_main_reward'] = float(reward[0])
                main_rewards.append(float(reward[0]))
                main_actions.append(action.copy())
                previous = action.copy()
                main_fell = bool(terminated[0])
                if main_fell or bool(timeout[0]):
                    reason = '|'.join((name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[0])))
                    break
                observation_history.append(from_obs(obs, previous, names))
                command_history.append(get_cmd(env))
            if len(main_rewards) == 1000 and (not main_fell):
                controller.set_fut_cmd(get_cmd(env))
                controller.update(from_obs(obs, previous, names))
                snapshots.append({'step': 1000, 'env': capture_env(env), 'controller': capture_controller(controller, previous, obs, observation_history[-25:], command_history[-25:]), 'next_main_action': None, 'next_main_reward': None})
            episode_length = len(main_rewards)
            for snap in snapshots:
                snap['episode_length'] = episode_length
                snap['fall'] = main_fell
            np.savez_compressed(out / 'main_episode.npz', actions=np.asarray(main_actions), rewards=np.asarray(main_rewards))
            selected = snapshots[:args.max_snapshots] if args.max_snapshots else snapshots
            all_rows = []
            for snap in selected:
                step_out = out / f"snapshot_{snap['step']:04d}"
                step_out.mkdir(exist_ok=True)
                export_snapshot(step_out / 'state.pt', snap['env'], snap['controller'])
                all_rows.extend(process_snapshot(env, controller, expert, critic, names, snap, step_out, seed, args.condition, args.max_candidates))
            write_csv(out / 'all_candidates.csv', all_rows)
            summary = {'condition': args.condition, 'seed': seed, 'mass_added_kg': mass, 'model_checkpoint': str(checkpoint.resolve()), 'model_step': model_step, 'physical_readback': physical, 'main_episode_length': episode_length, 'main_fell': main_fell, 'main_end_reason': reason, 'main_return': float(sum(main_rewards)), 'warmup_terminations': warmup_terminations, 'snapshots_total': len(snapshots), 'snapshots_processed': len(selected), 'candidate_count_per_snapshot': args.max_candidates or 66, 'duration_seconds': time.monotonic() - seed_started}
            (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print('SEED_SUMMARY', json.dumps(summary), flush=True)
        finally:
            env.close()
            omni.usd.get_context().new_stage()
    app.close()
if __name__ == '__main__':
    main()
