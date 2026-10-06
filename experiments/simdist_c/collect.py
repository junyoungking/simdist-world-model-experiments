"""Standalone stationary Go2 collection. No import of the released CLI script."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path
from common import ROOT, REPO, SOURCE, config, write_json, hashes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--friction', type=float, required=True)
    parser.add_argument('--added-mass', type=float, default=0.0)
    parser.add_argument('--forward-vel', type=float, default=None, help='Optional fixed forward velocity command (m/s); preserves released default when omitted.')
    parser.add_argument('--family', choices=['friction', 'added_mass', 'kp'], default=None, help='Declared one-axis intervention.')
    parser.add_argument('--kp', type=float)
    parser.add_argument('--kd', type=float)
    parser.add_argument('--gain-reset-probe', action='store_true')
    parser.add_argument('--forced-reset-audit-count', type=int, default=0, help='Force reset events before rollout; each is read back without mutation.')
    parser.add_argument('--checkpoint', default=str(SOURCE), help='Frozen controller/world-model checkpoint used for rollout.')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--role', choices=['smoke', 'eval', 'collection', 'control'], required=True)
    parser.add_argument('--log-dir', type=Path, default=None, help='Optional visualization run directory. Disabled by default.')
    parser.add_argument('--planner-score-log-dir', type=Path, default=None, help='Diagnostic-only MPPI score snapshots. Does not alter planner sampling or action selection.')
    args = parser.parse_args()
    assert (args.kp is None) == (args.kd is None), 'Kp and Kd must be specified together'
    if args.kp is None:
        args.kp, args.kd = (25.0, 0.5)
    if args.family is None:
        args.family = 'kp' if args.kp != 25.0 else 'added_mass' if args.added_mass else 'friction'
    assert args.family == 'kp' or (args.kp == 25.0 and args.kd == 0.5), 'Friction/added-mass target B must use fixed Kp=25, Kd=0.5'
    out = ROOT / 'datasets' / args.name
    out.mkdir(parents=True, exist_ok=False)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'
    from simdist.utils.jax import configure_jax_compilation_cache
    configure_jax_compilation_cache()
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    app = launcher.app
    import numpy as np
    import torch
    import h5py
    import jax
    from isaaclab.envs import ManagerBasedRLEnv
    from simdist.rl.go2 import Go2SimEnvCfg
    from simdist.utils import model as model_utils, config as config_utils
    from simdist.control.mppi import MppiController
    from omegaconf import OmegaConf
    writer = None
    log_dir = args.log_dir.resolve() if args.log_dir is not None else None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        if log_dir.exists():
            raise FileExistsError(f'Visualization log directory must be fresh: {log_dir}')
        (log_dir / 'raw').mkdir(parents=True)
        writer = SummaryWriter(str(log_dir / 'tensorboard'))
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=simdist_root(), text=True).strip()
        except Exception:
            git_hash = None
        (log_dir / 'metadata.json').write_text(json.dumps({'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'git_commit': git_hash, 'command': ' '.join(__import__('sys').argv), 'config_path': str(simdist_root() / 'config/simulate_go2.yaml'), 'environment_task': 'Go2SimEnvCfg', 'dataset_identifier': args.name, 'seed': args.seed, 'shift_family': args.family, 'shift_value': args.kp if args.family == 'kp' else args.added_mass if args.family == 'added_mass' else args.friction, 'source_condition': {'friction': 0.8, 'added_mass_kg': 0.0, 'kp': 25.0, 'kd': 0.5}, 'target_condition': {'friction': args.friction, 'added_mass_kg': args.added_mass, 'kp': args.kp, 'kd': args.kd}, 'checkpoint_path': str(resolve_artifact_path(args.checkpoint).resolve()), 'raw_schema': 'rollout_episode_<id>.npz contains pre_step_observation, action, reward_t, cumulative_reward_t.'}, indent=2) + '\n')
    planner_log_dir = args.planner_score_log_dir.resolve() if args.planner_score_log_dir is not None else None
    if planner_log_dir is not None:
        if planner_log_dir.exists():
            raise FileExistsError(f'Planner-score log directory must be fresh: {planner_log_dir}')
        planner_log_dir.mkdir(parents=True)
    cfg = config('simulate_go2')
    if args.forward_vel is not None:
        cfg['task']['forward_vel'] = args.forward_vel
    cfg['sim'].update(friction=args.friction, add_mass=args.added_mass, seed=args.seed, total_steps=args.steps, kp=args.kp, kd=args.kd, intervention_family=args.family)
    cfg['control']['seed'] = args.seed
    write_json(out / 'config.json', cfg)
    checkpoint = resolve_artifact_path(args.checkpoint).resolve()
    model, model_cfg, step = model_utils.load_model_from_ckpt(str(checkpoint))
    controller = MppiController(model, model_cfg, cfg['control'])
    names = config_utils.proprio_obs_names_from_sys_config(model_cfg['system'])
    env_cfg = Go2SimEnvCfg()
    env_cfg.seed = args.seed
    for key in ['forward_vel', 'kx', 'ky', 'k_heading']:
        setattr(env_cfg.commands.base_velocity, key, cfg['task'][key])
    generator = env_cfg.scene.terrain.terrain_generator
    generator.seed = args.seed
    generator.curriculum = False
    generator.size = (15.0, 15.0)
    generator.num_rows = generator.num_cols = 1
    generator.difficulty_range = (cfg['sim']['terrain_difficulty'],) * 2
    generator.sub_terrains = {cfg['sim']['terrain']: generator.sub_terrains[cfg['sim']['terrain']]}
    env_cfg.events.reset_base.params['pose_range']['x'] = (0.0, 0.0)
    material = env_cfg.events.physics_material.params
    material['static_friction_range'] = material['dynamic_friction_range'] = (args.friction,) * 2
    material['restitution_range'] = (cfg['sim']['restitution'],) * 2
    env_cfg.events.add_base_mass.params['mass_distribution_params'] = (cfg['sim']['add_mass'],) * 2
    gain_event = env_cfg.events.joint_stiffness_and_damping
    assert gain_event.mode == 'reset' and gain_event.params['operation'] == 'scale'
    env_cfg.events.joint_stiffness_and_damping = None
    motor_cfg = env_cfg.scene.robot.actuators['base_legs']
    env_cfg.scene.robot = env_cfg.scene.robot.replace(actuators={'base_legs': motor_cfg.replace(stiffness=args.kp, damping=args.kd)})
    fixed_gain = True
    import pprint
    (out / 'effective_env.txt').write_text(pprint.pformat(env_cfg.to_dict(), sort_dicts=True))
    env = ManagerBasedRLEnv(env_cfg)
    obs, _ = env.reset()
    zero_action = np.zeros(12, dtype=np.float32)
    zero_cmd = np.zeros(3, dtype=np.float32)
    gain_records = []
    actuator_trace = []

    def enforce_and_verify_gains(reason):
        robot = env.scene['robot']
        actuator = robot.actuators['base_legs']
        assert type(actuator).__name__ == 'DCMotor'
        ids = list(range(robot.num_joints)) if isinstance(actuator.joint_indices, slice) else list(actuator.joint_indices)
        assert len(ids) == 12 and ids == list(range(12)), ids
        as_numpy = lambda value: value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
        stiffness = as_numpy(actuator.stiffness)
        damping = as_numpy(actuator.damping)
        effort = as_numpy(actuator.effort_limit)
        saturation = as_numpy(actuator._saturation_effort)
        velocity = as_numpy(actuator.velocity_limit)
        if saturation.ndim == 0:
            saturation = np.full_like(effort, saturation)
        if fixed_gain:
            np.testing.assert_allclose(stiffness, args.kp, rtol=0, atol=1e-06)
            np.testing.assert_allclose(damping, args.kd, rtol=0, atol=1e-06)
        np.testing.assert_allclose(effort, 23.5, rtol=0, atol=1e-06)
        np.testing.assert_allclose(saturation, 23.5, rtol=0, atol=1e-06)
        np.testing.assert_allclose(velocity, 30.0, rtol=0, atol=1e-06)
        gain_records.append({'reason': reason, 'kp': stiffness.tolist(), 'kd': damping.tolist(), 'effort_limit': effort.tolist(), 'saturation_effort': saturation.tolist(), 'velocity_limit': velocity.tolist(), 'actuator_type': type(actuator).__name__, 'fixed_gain_protocol': fixed_gain})
    enforce_and_verify_gains('initial_reset')
    if args.gain_reset_probe:
        obs, _ = env.reset()
        enforce_and_verify_gains('explicit_reset_probe')
    for reset_index in range(args.forced_reset_audit_count):
        obs, _ = env.reset()
        enforce_and_verify_gains(f'forced_reset_{reset_index + 1:02d}')

    def get_obs(ob):
        prop = np.concatenate([ob['obs'][n].squeeze().cpu().numpy() for n in names]).astype(np.float32)
        ext = ob['obs']['height_scan'].squeeze().cpu().numpy().astype(np.float32)
        assert prop.shape == (35,) and ext.shape == (315,)
        assert np.isfinite(prop).all() and np.isfinite(ext).all(), 'Nonfinite observations'
        return (prop, ext)

    def make_x(ob, action):
        p, e = get_obs(ob)
        return {'proprio_obs': p, 'extero_obs': e, 'prev_action': action}
    controller.initialize(make_x(obs, zero_action), zero_cmd)
    controller.reset(make_x(obs, zero_action), zero_cmd)
    controller.set_fut_cmd(zero_cmd)
    raw = out / 'raw'
    raw.mkdir()
    rows, episodes, tracking = ([], [], [])
    local_step, total, previous = (0, 0, zero_action.copy())
    planner_control_step = 0
    planner_recent = []
    planner_logged_steps = set()
    start_time = time.monotonic()
    materials = env.scene['robot'].root_physx_view.get_material_properties().cpu().numpy()
    np.save(out / 'robot_material_properties.npy', materials)
    from simdist.rl.go2_mdp import randomize_rigid_body_material_same
    event = next((t.func for t in env.event_manager._mode_term_cfgs['startup'] if isinstance(t.func, randomize_rigid_body_material_same)))
    foot_shapes = []
    for body_id in event.asset_cfg.body_ids:
        first = sum(event.num_shapes_per_body[:body_id])
        foot_shapes.extend(range(first, first + event.num_shapes_per_body[body_id]))
    assert foot_shapes
    np.testing.assert_allclose(materials[:, foot_shapes, :2], args.friction, atol=1e-06)
    write_json(out / 'physics_verified.json', {'foot_shape_indices': foot_shapes, 'foot_materials': materials[:, foot_shapes].tolist(), 'scope': 'Released material event changes the four feet only; other bodies unchanged.'})
    robot = env.scene['robot']
    mass_term = env.event_manager.get_term_cfg('add_base_mass')
    ids = list(mass_term.params['asset_cfg'].body_ids)
    assert len(ids) == 1 and robot.body_names[ids[0]] == 'base'
    assert mass_term.mode == 'startup' and mass_term.params['operation'] == 'add'
    assert mass_term.params.get('recompute_inertia', True)
    default_mass = robot.data.default_mass.cpu().numpy()
    actual_mass = robot.root_physx_view.get_masses().cpu().numpy()
    default_inertia = robot.data.default_inertia.cpu().numpy()
    actual_inertia = robot.root_physx_view.get_inertias().cpu().numpy()
    expected_mass = default_mass.copy()
    expected_mass[:, ids] += args.added_mass
    expected_inertia = default_inertia.copy()
    expected_inertia[:, ids] *= (expected_mass[:, ids] / default_mass[:, ids])[..., None]
    np.testing.assert_allclose(actual_mass, expected_mass, rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(actual_inertia, expected_inertia, rtol=2e-05, atol=1e-07)
    write_json(out / 'mass_verified.json', {'requested_additive_mass_kg': args.added_mass, 'body_ids': ids, 'body_names': [robot.body_names[i] for i in ids], 'default_base_mass_kg': default_mass[:, ids].tolist(), 'effective_base_mass_kg': actual_mass[:, ids].tolist(), 'default_base_inertia': default_inertia[:, ids].tolist(), 'effective_base_inertia': actual_inertia[:, ids].tolist(), 'expected_base_inertia': expected_inertia[:, ids].tolist(), 'all_body_mass_and_inertia_match_released_semantics': True, 'operation': 'add', 'timing': 'startup', 'recompute_inertia': True})

    def finish(reason):
        nonlocal rows
        if not rows:
            return
        idx = len(episodes)
        with h5py.File(raw / f'episode_{idx:04d}.hdf5', 'x') as f:
            for k in ['time', 'proprio_obs', 'extero_obs', 'actions', 'commands', 'reward']:
                f.create_dataset(k, data=np.asarray([r[k] for r in rows], dtype=np.int64 if k == 'time' else np.float32))
        if log_dir is not None:
            rewards = np.asarray([r['reward'] for r in rows], dtype=np.float32)
            np.savez_compressed(log_dir / 'raw' / f'rollout_episode_{idx:04d}.npz', timestep=np.asarray([r['global_step'] for r in rows], dtype=np.int64), episode_id=np.full(len(rows), idx, dtype=np.int64), seed=np.full(len(rows), args.seed, dtype=np.int64), reward_t=rewards, cumulative_reward_t=np.cumsum(rewards, dtype=np.float64), proprio_obs=np.asarray([r['proprio_obs'] for r in rows], dtype=np.float32), extero_obs=np.asarray([r['extero_obs'] for r in rows], dtype=np.float32), action=np.asarray([r['actions'] for r in rows], dtype=np.float32), command=np.asarray([r['commands'] for r in rows], dtype=np.float32), proprio_feature_names=np.asarray(names), action_feature_names=np.asarray([f'action_{i}' for i in range(12)]))
        episodes.append({'index': idx, 'length': len(rows), 'end_reason': reason, 'stabilization_rows': min(len(rows), cfg['sim']['reset_steps']), 'usable_observed_transitions': max(0, len(rows) - 1)})
        rows = []

    def write_planner_score_snapshot(entry, reason):
        """Persist a fixed candidate subset without changing the MPPI state/RNG."""
        if planner_log_dir is None or entry['planner_control_step'] in planner_logged_steps:
            return
        base_actions = controller._get_base_policy_actions(entry['model_inputs'])
        acts, rewards, values, total_scores = controller.score_snapshot(entry['pre_mppi_state'], base_actions, entry['model_inputs'])
        acts = np.asarray(acts)
        rewards = np.asarray(rewards)
        values = np.asarray(values)
        total_scores = np.asarray(total_scores)
        reward_sums = np.sum(rewards * np.asarray(controller.discounts)[None, :], axis=-1)
        terminal_values = values[:, -1]
        terminal_contrib = terminal_values * float(np.asarray(controller.final_discount))
        np.testing.assert_allclose(total_scores, reward_sums + terminal_contrib, rtol=2e-05, atol=1e-06)
        order = np.argsort(-total_scores, kind='stable')
        ranks = np.empty(len(order), dtype=np.int64)
        ranks[order] = np.arange(1, len(order) + 1)
        middle_start = max(0, len(order) // 2 - 5)
        middle = order[middle_start:middle_start + 10]
        local_rng = np.random.default_rng(args.seed * 1000003 + entry['planner_control_step'])
        random_ids = local_rng.choice(len(order), size=min(10, len(order)), replace=False)
        labels = {}
        for label, ids in [('top10', order[:10]), ('middle10', middle), ('fixed_random10', random_ids)]:
            for candidate_id in ids:
                labels.setdefault(int(candidate_id), []).append(label)
        retained = np.asarray(sorted(labels), dtype=np.int64)
        filename = planner_log_dir / f"step_{entry['planner_control_step']:05d}_global_{entry['global_step']:05d}.npz"
        np.savez_compressed(filename, planner_control_step=np.int64(entry['planner_control_step']), global_step=np.int64(entry['global_step']), command=np.asarray(entry['command'], dtype=np.float32), retained_candidate_ids=retained, candidate_rank=ranks[retained], candidate_subset_labels=np.asarray(['|'.join(labels[int(i)]) for i in retained]), candidate_actions=acts[retained], predicted_reward_sequence=rewards[retained], predicted_terminal_value=terminal_values[retained], discounted_reward_sum=reward_sums[retained], terminal_value_contribution=terminal_contrib[retained], total_mppi_score=total_scores[retained], selected_candidate_id=np.int64(-1), selected_candidate_flag=np.zeros(len(retained), dtype=bool), planner_output_is_weighted_elite_mean=np.bool_(True), executed_action=np.asarray(entry['executed_action'], dtype=np.float32), reason=np.asarray(reason))
        planner_logged_steps.add(entry['planner_control_step'])
    try:
        while total < args.steps and app.is_running():
            resetting = local_step < cfg['sim']['reset_steps']
            cmd = zero_cmd if resetting else env.command_manager.get_command('base_velocity').cpu().numpy()[0].copy()
            controller.set_fut_cmd(cmd)
            controller.update(make_x(obs, previous))
            inputs = controller._make_model_inputs(controller.dummy_actions)
            assert all((np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(inputs))), 'Nonfinite model input'
            pre_mppi_state = controller.mppi_state
            output_actions = controller.run_control()['actions']
            assert np.isfinite(output_actions).all(), 'Nonfinite controller action'
            action = zero_action.copy() if resetting else output_actions[0].astype(np.float32).copy()
            score_entry = None
            if not resetting:
                score_entry = {'planner_control_step': planner_control_step, 'global_step': total, 'command': cmd.copy(), 'pre_mppi_state': pre_mppi_state, 'model_inputs': inputs, 'executed_action': action.copy()}
                planner_recent.append(score_entry)
                planner_recent = planner_recent[-10:]
                if planner_log_dir is not None and (planner_control_step < 10 or planner_control_step % 25 == 0):
                    write_planner_score_snapshot(score_entry, 'predeclared_first10_or_every25')
                planner_control_step += 1
            prop, ext = get_obs(obs)
            rows.append({'time': round(total * env.step_dt * 1000000000.0), 'proprio_obs': prop.copy(), 'extero_obs': ext.copy(), 'actions': action, 'commands': cmd.copy(), 'global_step': total})
            if not resetting:
                velocity = env.scene['robot'].data.root_lin_vel_b[0].cpu().numpy()
                tracking.append((velocity[:2] - cmd[:2]).tolist())
            obs, reward, terminated, truncated, extras = env.step(torch.from_numpy(action).unsqueeze(0).to(env.device))
            reward_value = float(reward[0].detach().cpu().item())
            assert np.isfinite(reward_value), 'Nonfinite environment reward'
            rows[-1]['reward'] = reward_value
            if writer is not None:
                writer.add_scalar('rollout/reward_step', reward_value, total)
                writer.add_scalar('rollout/cumulative_reward', sum((r['reward'] for r in rows)), total)
                writer.add_scalar('rollout/action_norm', float(np.linalg.norm(action)), total)
            get_obs(obs)
            if fixed_gain:
                trace = {'time': round(total * env.step_dt * 1000000000.0), 'joint_pos': robot.data.joint_pos[0].detach().cpu().numpy().copy(), 'joint_pos_target': robot.data.joint_pos_target[0].detach().cpu().numpy().copy(), 'joint_vel': robot.data.joint_vel[0].detach().cpu().numpy().copy(), 'computed_torque': robot.data.computed_torque[0].detach().cpu().numpy().copy(), 'applied_torque': robot.data.applied_torque[0].detach().cpu().numpy().copy()}
                assert all((np.isfinite(v).all() for k, v in trace.items() if k != 'time')), 'Nonfinite actuator trace'
                actuator_trace.append(trace)
            total += 1
            local_step += 1
            previous = action
            write_json(out / 'progress.json', {'completed_steps': total})
            if bool(terminated[0]) or bool(truncated[0]):
                if planner_log_dir is not None:
                    for recent_entry in planner_recent:
                        write_planner_score_snapshot(recent_entry, 'termination_adjacent_preceding10')
                finish('terminated' if bool(terminated[0]) else 'timeout')
                enforce_and_verify_gains('post_reset')
                local_step = 0
                previous = zero_action.copy()
                controller.reset(make_x(obs, previous), zero_cmd)
            if total % 100 == 0:
                print(f'COLLECTION_PROGRESS {total}/{args.steps}', flush=True)
        finish('budget_censored')
        assert total == args.steps, 'Requested steps did not complete'
        if fixed_gain:
            trace_path = out / 'actuator_trace.npz'
            np.savez_compressed(trace_path, **{k: np.asarray([r[k] for r in actuator_trace]) for k in actuator_trace[0]})
            computed = np.asarray([r['computed_torque'] for r in actuator_trace])
            applied = np.asarray([r['applied_torque'] for r in actuator_trace])
            saturation_fraction = float(np.mean(np.abs(computed - applied) > 1e-05))
            write_json(out / 'effective_gain_audit.json', {'fixed_gain_protocol': True, 'requested_kp': args.kp, 'requested_kd': args.kd, 'controlled_joint_count': 12, 'joint_names': robot.joint_names, 'reset_readbacks': gain_records, 'random_gain_event_disabled': True, 'trace_file': str(trace_path), 'trace_sha256': __import__('hashlib').sha256(trace_path.read_bytes()).hexdigest(), 'saturation_fraction': saturation_fraction, 'computed_torque_rms': float(np.sqrt(np.mean(computed ** 2))), 'applied_torque_rms': float(np.sqrt(np.mean(applied ** 2))), 'computed_torque_peak': float(np.max(np.abs(computed))), 'applied_torque_peak': float(np.max(np.abs(applied)))})
        write_json(out / 'actuator_readback.json', {'default_nominal_kp': 25.0, 'default_nominal_kd': 0.5, 'reset_randomization_active': not fixed_gain, 'declared_intervention_family': args.family, 'records': gain_records, 'constant_effort_limit': 23.5, 'constant_velocity_limit': 30.0})
        write_json(out / 'rollout.json', {'name': args.name, 'role': args.role, 'seed': args.seed, 'intervention_family': args.family, 'friction': args.friction, 'added_mass_kg': args.added_mass, 'kp': args.kp if args.family == 'kp' else None, 'kd': args.kd if args.family == 'kp' else None, 'fixed_kp': args.kp, 'fixed_kd': args.kd, 'requested_steps': args.steps, 'completed_steps': total, 'episodes': episodes, 'raw_transition_count': sum((e['usable_observed_transitions'] for e in episodes)), 'nominal_windows': max(0, args.steps - 35), 'episode_split_nominal_windows': sum((max(0, e['length'] - 35) for e in episodes)), 'median_episode_length': float(np.median([e['length'] for e in episodes])), 'episode_length_censoring': 'Includes budget-censored segment; same rule for A0/B', 'tracking_rmse_xy': np.sqrt(np.mean(np.square(tracking), axis=0)).tolist() if tracking else None, 'loop_wall_seconds': time.monotonic() - start_time, 'source_checkpoint': str(checkpoint), 'source_checkpoint_step': step, 'source_checkpoint_hashes': hashes(checkpoint), 'raw_hashes': hashes(raw), 'finite': True, 'actuator_readback_path': str(out / 'actuator_readback.json'), 'planner_score_instrumentation': {'enabled': planner_log_dir is not None, 'path': str(planner_log_dir) if planner_log_dir is not None else None, 'schedule': 'first 10 non-reset planner steps; every 25th thereafter; preceding 10 at termination when available', 'candidate_selection': 'top10, middle10, fixed local-NumPy-random10 from final MPPI iteration', 'planner_rng_mutated': False, 'selected_candidate_semantics': 'none: executed action is the final weighted-elite mean, not a sampled candidate'}})
    except BaseException as exc:
        import traceback
        (out / 'failure.txt').write_text(traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        env.close()
        app.close()
if __name__ == '__main__':
    main()
