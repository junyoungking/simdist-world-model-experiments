"""Canonical fixed +8kg/0kg configuration and read-only runtime checks."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
ROOT = artifact_dir('privileged_ppo')
REPO = PORTFOLIO_ROOT

def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, slice):
        return 'slice' + str((value.start, value.stop, value.step))
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    return str(value)

def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def write(path, value):
    path = resolve_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')

def configure(seed, num_envs=1, added_mass=8.0):
    from simdist.rl.go2 import Go2SimEnvCfg
    cfg = Go2SimEnvCfg()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    command = cfg.commands.base_velocity
    for name, value in [('forward_vel', 0.5), ('kx', 1.0), ('ky', 1.0), ('k_heading', 1.0)]:
        setattr(command, name, value)
    gen = cfg.scene.terrain.terrain_generator
    gen.seed = seed
    gen.curriculum = False
    gen.size = (15.0, 15.0)
    gen.num_rows = gen.num_cols = 1
    gen.difficulty_range = (0.4, 0.4)
    gen.sub_terrains = {'boxes': gen.sub_terrains['boxes']}
    cfg.events.reset_base.params['pose_range']['x'] = (0.0, 0.0)
    material = cfg.events.physics_material.params
    material['static_friction_range'] = material['dynamic_friction_range'] = (0.8, 0.8)
    material['restitution_range'] = (0.05, 0.05)
    cfg.events.add_base_mass.params['mass_distribution_params'] = (added_mass, added_mass)
    cfg.events.joint_stiffness_and_damping = None
    motor = cfg.scene.robot.actuators['base_legs']
    cfg.scene.robot = cfg.scene.robot.replace(actuators={'base_legs': motor.replace(stiffness=25.0, damping=0.5)})
    historical = json.loads(document_path('privileged_ppo', 'HISTORICAL_ENV_CONTRACT.json').read_text())
    current = plain(cfg.to_dict())
    expected = json.loads(json.dumps(historical))
    expected['events']['add_base_mass']['params']['mass_distribution_params'] = [added_mass, added_mass]
    for name in ['actions', 'commands', 'rewards', 'terminations', 'events']:
        assert current[name] == expected[name], f'Canonical config mismatch: {name}'
    assert cfg.sim.dt == 0.005 and cfg.decimation == 4
    return cfg

def runtime_readback(env, added_mass, reason, include_obs=True):
    robot = env.scene['robot']
    actuator = robot.actuators['base_legs']
    term = env.event_manager.get_term_cfg('add_base_mass')
    ids = list(term.params['asset_cfg'].body_ids)
    assert ids == [0] and robot.body_names[0] == 'base'
    assert term.mode == 'startup' and term.params['operation'] == 'add'
    expected = robot.data.default_mass.cpu().numpy().copy()
    expected[:, ids] += added_mass
    actual = robot.root_physx_view.get_masses().cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-06)
    original_inertia = robot.data.default_inertia.cpu().numpy()
    expected_inertia = original_inertia.copy()
    expected_inertia[:, ids] *= (expected[:, ids] / robot.data.default_mass.cpu().numpy()[:, ids])[..., None]
    np.testing.assert_allclose(robot.root_physx_view.get_inertias().cpu().numpy(), expected_inertia, rtol=2e-05, atol=1e-07)
    np.testing.assert_allclose(actuator.stiffness.cpu().numpy(), 25.0, rtol=0, atol=1e-06)
    np.testing.assert_allclose(actuator.damping.cpu().numpy(), 0.5, rtol=0, atol=1e-06)
    np.testing.assert_allclose(actuator.effort_limit.cpu().numpy(), 23.5, rtol=0, atol=1e-06)
    np.testing.assert_allclose(actuator.velocity_limit.cpu().numpy(), 30.0, rtol=0, atol=1e-06)
    from simdist.rl.go2_mdp import randomize_rigid_body_material_same
    material_term = next((t.func for t in env.event_manager._mode_term_cfgs['startup'] if isinstance(t.func, randomize_rigid_body_material_same)))
    shapes = []
    for body_id in material_term.asset_cfg.body_ids:
        start = sum(material_term.num_shapes_per_body[:body_id])
        shapes.extend(range(start, start + material_term.num_shapes_per_body[body_id]))
    materials = robot.root_physx_view.get_material_properties().cpu().numpy()
    feet = materials[:, shapes]
    np.testing.assert_allclose(feet[..., :2], 0.8, rtol=0, atol=1e-06)
    np.testing.assert_allclose(feet[..., 2], 0.05, rtol=0, atol=1e-06)
    state = {name: getattr(robot.data, name).detach().cpu().numpy() for name in ['root_state_w', 'joint_pos', 'joint_vel']}
    initial = hashlib.sha256()
    for name, array in state.items():
        initial.update(name.encode())
        initial.update(np.ascontiguousarray(array).tobytes())
    mesh = env.scene.terrain.meshes['terrain']
    terrain_sha = hashlib.sha256(np.asarray(mesh.vertices).tobytes() + np.asarray(mesh.faces).tobytes()).hexdigest()
    joint_friction = robot.root_physx_view.get_dof_friction_coefficients().cpu().numpy()
    result = {'reason': reason, 'num_envs': env.num_envs, 'base_mass_min': float(actual[:, 0].min()), 'base_mass_max': float(actual[:, 0].max()), 'added_mass': added_mass, 'default_base_mass': float(robot.data.default_mass[0, 0]), 'mass_inertia_PASS': True, 'kp': [float(actuator.stiffness.min()), float(actuator.stiffness.max())], 'kd': [float(actuator.damping.min()), float(actuator.damping.max())], 'effort_limit': [float(actuator.effort_limit.min()), float(actuator.effort_limit.max())], 'velocity_limit': [float(actuator.velocity_limit.min()), float(actuator.velocity_limit.max())], 'joint_friction_first_env': joint_friction[0].tolist(), 'joint_friction_sha256': hashlib.sha256(np.ascontiguousarray(joint_friction).tobytes()).hexdigest(), 'foot_material_min': feet.reshape(-1, 3).min(axis=0).tolist(), 'foot_material_max': feet.reshape(-1, 3).max(axis=0).tolist(), 'control_dt': env.step_dt, 'physics_dt': env.cfg.sim.dt, 'command': env.command_manager.get_command('base_velocity').detach().cpu().tolist(), 'reference_vx': env.cfg.commands.base_velocity.forward_vel, 'initial_root_joint_state_sha256': initial.hexdigest(), 'terrain_mesh_sha256': terrain_sha, 'reward_signature': digest(env.cfg.to_dict()['rewards']), 'termination_signature': digest(env.cfg.to_dict()['terminations']), 'action_signature': digest(env.cfg.to_dict()['actions']), 'command_signature': digest(env.cfg.to_dict()['commands'])}
    if include_obs:
        obs = env.observation_manager.compute()
        policy = obs['policy']
        assert policy.shape == (env.num_envs, 433) and torch.isfinite(policy).all()
        result['policy_obs_shape'] = list(policy.shape)
        result['policy_obs_ranges'] = {}
        offset = 0
        names = env.observation_manager.active_terms['policy']
        dims = env.observation_manager.group_obs_term_dim['policy']
        for name, dim in zip(names, dims):
            n = int(np.prod(dim))
            values = policy[:, offset:offset + n]
            result['policy_obs_ranges'][name] = {'start': offset, 'size': n, 'min': float(values.min()), 'max': float(values.max())}
            offset += n
        assert offset == 433
    return result
