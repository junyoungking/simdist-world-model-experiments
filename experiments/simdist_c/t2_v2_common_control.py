"""T2-v2: normal common-control N=2 synchronization and pairwise branches.

This is deliberately separate from the historical zero-action-prefix T2.
"""
from __future__ import annotations
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import argparse, csv, hashlib, json, os, subprocess, sys, traceback
from pathlib import Path
import numpy as np
REPO = PORTFOLIO_ROOT
WORK = PORTFOLIO_ROOT
OUT = legacy_dir('12_t2_v2_common_control_pairwise')
OLD = legacy_dir('research_topic_validation_20260913T173407Z')
TIMES, SMOKE_SEEDS = ((10, 50, 100), (13302, 13303))
CONTROL_SEEDS = {1: 7101, 2: 7102, 3: 7103}
PAIRS = ((1, 472, 'large'), (1, 354, 'large'), (1, 236, 'large'), (118, 472, 'large'), (60, 236, 'medium'), (118, 354, 'medium'), (177, 354, 'medium'), (236, 472, 'medium'), (1, 20, 'small'), (118, 138, 'small'), (236, 256, 'small'), (354, 374, 'small'))
FIELDS = ('root_position_local', 'root_orientation', 'root_linear_velocity', 'root_angular_velocity', 'joint_positions', 'joint_velocities', 'proprio_observation', 'extero_observation', 'projected_gravity', 'phase', 'command', 'episode_step', 'termination_flag', 'joint_position_target', 'computed_torque', 'applied_torque')
TOL = 1e-05

def args():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('sync', 'branches'))
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--condition', choices=('CUMULATIVE', 'RECENT'))
    p.add_argument('--seed', type=int)
    return p.parse_args()

def wc(p, rows, fields):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def append_wc(p, rows, fields, identity):
    old = []
    if p.exists():
        with p.open(newline='') as f:
            old = list(csv.DictReader(f))
    kept = [r for r in old if tuple((r.get(k, '') for k in identity)) not in {tuple((str(x.get(k, '')) for k in identity)) for x in rows}]
    wc(p, kept + rows, fields)

def gpu():
    return subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()

def sha_actions(actions):
    return hashlib.sha256(np.asarray(actions, dtype=np.float32).tobytes()).hexdigest()

def metrics(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    flat = np.abs(d).reshape(-1)
    return (float(flat.max()), float(flat.mean()), float(np.linalg.norm(d.reshape(-1))), int(flat.size))

def setup(seed, model_path):
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.10'
    from isaaclab.envs import ManagerBasedRLEnv
    from simdist.rl.go2 import Go2SimEnvCfg
    from simdist.utils import model as model_utils, config as config_utils
    from simdist.control.mppi import MppiController
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    import torch
    with initialize_config_dir(config_dir=str(simdist_root() / 'config'), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name='simulate_go2'), resolve=True)
    cfg['task']['forward_vel'] = 0.5
    cfg['sim'].update(friction=0.8, add_mass=8.0, kp=25.0, kd=0.5, intervention_family='added_mass')
    cfg['control']['seed'] = seed
    ec = Go2SimEnvCfg()
    ec.scene.num_envs = 2
    ec.seed = seed
    for k in ('forward_vel', 'kx', 'ky', 'k_heading'):
        setattr(ec.commands.base_velocity, k, cfg['task'][k])
    tr = ec.scene.terrain.terrain_generator
    tr.seed, tr.curriculum = (seed, False)
    tr.size = (15.0, 15.0)
    tr.num_rows = tr.num_cols = 1
    tr.difficulty_range = (cfg['sim']['terrain_difficulty'],) * 2
    tr.sub_terrains = {cfg['sim']['terrain']: tr.sub_terrains[cfg['sim']['terrain']]}
    ec.events.reset_base.params['pose_range']['x'] = (0.0, 0.0)
    mat = ec.events.physics_material.params
    mat['static_friction_range'] = mat['dynamic_friction_range'] = (0.8, 0.8)
    mat['restitution_range'] = (cfg['sim']['restitution'],) * 2
    ec.events.add_base_mass.params['mass_distribution_params'] = (8.0, 8.0)
    ec.events.joint_stiffness_and_damping = None
    ec.events.joint_friction = None
    ec.events.base_external_force_torque = None
    motor = ec.scene.robot.actuators['base_legs']
    ec.scene.robot = ec.scene.robot.replace(actuators={'base_legs': motor.replace(stiffness=25.0, damping=0.5)})
    env = ManagerBasedRLEnv(ec)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)
    obs, _ = env.reset()
    model, mc, _ = model_utils.load_model_from_ckpt(str(model_path))
    names = config_utils.proprio_obs_names_from_sys_config(mc['system'])

    def oe(o):
        return (np.concatenate([o['obs'][n].detach().cpu().numpy() for n in names], axis=-1).astype(np.float32), o['obs']['height_scan'].detach().cpu().numpy().astype(np.float32))

    def x(o, prev):
        p, e = oe(o)
        return {'proprio_obs': p[0].copy(), 'extero_obs': e[0].copy(), 'prev_action': prev.copy()}
    ctl = MppiController(model, mc, cfg['control'])
    zero = np.zeros(12, np.float32)
    zc = np.zeros(3, np.float32)
    ctl.initialize(x(obs, zero), zc)
    ctl.reset(x(obs, zero), zc)
    return (env, obs, model, ctl, names, x, oe, zero)

def capture(env, obs, names, command, terminated):
    data = env.scene['robot'].data
    p = np.concatenate([obs['obs'][n].detach().cpu().numpy() for n in names], axis=-1).astype(np.float32)
    e = obs['obs']['height_scan'].detach().cpu().numpy().astype(np.float32)
    return {'root_position_local': data.root_pos_w.detach().cpu().numpy() - env.scene.env_origins.detach().cpu().numpy(), 'root_orientation': data.root_quat_w.detach().cpu().numpy(), 'root_linear_velocity': data.root_lin_vel_w.detach().cpu().numpy(), 'root_angular_velocity': data.root_ang_vel_w.detach().cpu().numpy(), 'joint_positions': data.joint_pos.detach().cpu().numpy(), 'joint_velocities': data.joint_vel.detach().cpu().numpy(), 'proprio_observation': p, 'extero_observation': e, 'projected_gravity': p[:, 9:12], 'phase': p[:, -2:], 'command': np.broadcast_to(command, (2, 3)).astype(np.float32).copy(), 'episode_step': env.episode_length_buf.detach().cpu().numpy()[:, None], 'termination_flag': np.asarray(terminated, np.float32)[:, None], 'joint_position_target': data.joint_pos_target.detach().cpu().numpy(), 'computed_torque': data.computed_torque.detach().cpu().numpy(), 'applied_torque': data.applied_torque.detach().cpu().numpy()}

def prefix(env, obs, ctl, names, x, zero, target_t):
    import torch
    prev = zero.copy()
    term = np.zeros(2, bool)
    actions = []
    states = {}
    for step in range(target_t + 1):
        command = env.command_manager.get_command('base_velocity').detach().cpu().numpy()[0].copy()
        state = capture(env, obs, names, command, term)
        if step in TIMES or step == target_t:
            states[step] = (state, command.copy(), ctl.mppi_state, prev.copy(), obs)
        if step == target_t:
            break
        ctl.set_fut_cmd(command)
        ctl.update(x(obs, prev))
        action = ctl.run_control()['actions'][0].astype(np.float32)
        actions.append(action.copy())
        obs, _, tt, tr, _ = env.step(torch.from_numpy(np.broadcast_to(action, (2, 12)).copy()).to(env.device))
        term |= tt.detach().cpu().numpy() | tr.detach().cpu().numpy()
        prev = action
        if term.any():
            return (obs, term, actions, states)
    return (obs, term, actions, states)

def sync(out, only_condition=None, only_seed=None):
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    raw = []
    gate = []
    reset = []
    hashes = []
    failure = None
    try:
        for kind in (only_condition,) if only_condition else ('CUMULATIVE', 'RECENT'):
            mp = OLD / 'models' / f'ns_r1_to_b_{kind.lower()}_K1000'
            for seed in (only_seed,) if only_seed is not None else SMOKE_SEEDS:
                env, obs, model, ctl, names, x, oe, zero = setup(seed, mp)
                cmd = env.command_manager.get_command('base_velocity').detach().cpu().numpy()[0].copy()
                initial = capture(env, obs, names, cmd, np.zeros(2, bool))
                reset_max = 0.0
                for f in FIELDS:
                    ma, me, l2, dim = metrics(initial[f][0], initial[f][1])
                    reset_max = max(reset_max, ma)
                    reset += [{'model_condition': kind, 'seed': seed, 'field': f, 'env_id': i, 'value_json': json.dumps(np.asarray(initial[f][i]).tolist()), 'max_abs_error': ma, 'mean_abs_error': me, 'l2_error': l2, 'field_dimension': dim} for i in range(2)]
                obs, term, acts, states = prefix(env, obs, ctl, names, x, zero, max(TIMES))
                for t in TIMES:
                    if t not in states:
                        gate.append({'model_condition': kind, 'seed': seed, 'planner_time': t, 'decision': 'PRECHECKPOINT_TERMINATION', 'alive_env0': False, 'alive_env1': False, 'action_history_sha256': sha_actions(acts), 'action_count': len(acts), 'largest_state_max_abs_error': '', 'tolerance_max_abs': TOL, 'reason': 'terminated before checkpoint'})
                        continue
                    state, command, _, _, _ = states[t]
                    largest = 0.0
                    for f in FIELDS:
                        ma, me, l2, dim = metrics(state[f][0], state[f][1])
                        largest = max(largest, ma)
                        for i in range(2):
                            raw.append({'model_condition': kind, 'seed': seed, 'planner_time': t, 'env_id': i, 'field': f, 'value_json': json.dumps(np.asarray(state[f][i]).tolist()), 'max_abs_error': ma, 'mean_abs_error': me, 'l2_error': l2, 'field_dimension': dim})
                    alive = not term.any()
                    dec = 'SYNC_PASS' if alive and largest <= TOL else 'PRECHECKPOINT_TERMINATION' if not alive else 'SYNC_FAIL'
                    gate.append({'model_condition': kind, 'seed': seed, 'planner_time': t, 'decision': dec, 'alive_env0': not bool(term[0]), 'alive_env1': not bool(term[1]), 'action_history_sha256': sha_actions(acts[:t]), 'action_count': min(t, len(acts)), 'largest_state_max_abs_error': largest, 'tolerance_max_abs': TOL, 'reason': ''})
                hashes.append({'model_condition': kind, 'seed': seed, 'action_count': len(acts), 'action_history_sha256': sha_actions(acts), 'identical_broadcast_actions': True, 'normal_common_control': True, 'reset_largest_max_abs_error': reset_max})
    except BaseException:
        failure = traceback.format_exc()
    finally:
        syncd = out / 'sync'
        append_wc(syncd / 'T2V2_RESET_SYNC_RAW.csv', reset, ['model_condition', 'seed', 'field', 'env_id', 'value_json', 'max_abs_error', 'mean_abs_error', 'l2_error', 'field_dimension'], ('model_condition', 'seed', 'field', 'env_id'))
        append_wc(syncd / 'T2V2_PREFIX_ACTION_HASHES.csv', hashes, ['model_condition', 'seed', 'action_count', 'action_history_sha256', 'identical_broadcast_actions', 'normal_common_control', 'reset_largest_max_abs_error'], ('model_condition', 'seed'))
        append_wc(syncd / 'T2V2_SYNC_RAW.csv', raw, ['model_condition', 'seed', 'planner_time', 'env_id', 'field', 'value_json', 'max_abs_error', 'mean_abs_error', 'l2_error', 'field_dimension'], ('model_condition', 'seed', 'planner_time', 'env_id', 'field'))
        append_wc(syncd / 'T2V2_SYNC_GATE.csv', gate, ['model_condition', 'seed', 'planner_time', 'decision', 'alive_env0', 'alive_env1', 'action_history_sha256', 'action_count', 'largest_state_max_abs_error', 'tolerance_max_abs', 'reason'], ('model_condition', 'seed', 'planner_time'))
        if failure:
            (syncd / 'T2V2_FAILURE.txt').write_text(failure)
        if app is not None:
            app.close()
    if failure:
        raise RuntimeError('T2-v2 sync failure; see T2V2_FAILURE.txt')
if __name__ == '__main__':
    a = args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.phase == 'sync':
        sync(a.out, a.condition, a.seed)
    else:
        raise NotImplementedError('Branches are enabled only after the recorded T2-v2 sync gate is reviewed.')
