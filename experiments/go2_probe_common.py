"""Scratch-only construction of the fixed-gain Go2 comparison environment."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
from pathlib import Path
ROOT = simdist_root()
MODEL = checkpoint_root() / 'models/go2_2026-02-12'
EXPERT = checkpoint_root() / 'rl/go2_2026-02-11/policies/policy_4999.pt'

def config(seed, mass, forward_vel=0.5, constant_command=False):
    from simdist.rl.go2 import Go2SimEnvCfg
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
    ec = Go2SimEnvCfg()
    ec.seed = seed
    ec.scene.num_envs = 1
    tr = ec.scene.terrain.terrain_generator
    tr.seed, tr.curriculum = (seed, False)
    tr.size = (15.0, 15.0)
    tr.num_rows = tr.num_cols = 1
    tr.difficulty_range = (0.4, 0.4)
    tr.sub_terrains = {'boxes': tr.sub_terrains['boxes']}
    ec.events.reset_base.params['pose_range']['x'] = (0.0, 0.0)
    mat = ec.events.physics_material.params
    mat['static_friction_range'] = (0.8, 0.8)
    mat['dynamic_friction_range'] = (0.8, 0.8)
    mat['restitution_range'] = (0.05, 0.05)
    ec.events.add_base_mass.params['mass_distribution_params'] = (mass, mass)
    ec.events.joint_stiffness_and_damping = None
    ec.events.joint_friction = None
    ec.events.base_external_force_torque = None
    motor = ec.scene.robot.actuators['base_legs']
    ec.scene.robot = ec.scene.robot.replace(actuators={'base_legs': motor.replace(stiffness=25.0, damping=0.5)})
    if constant_command:
        ec.commands.base_velocity = mdp.UniformVelocityCommandCfg(asset_name='robot', resampling_time_range=(1000.0, 1000.0), heading_command=False, rel_standing_envs=0.0, rel_heading_envs=0.0, debug_vis=False, ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(forward_vel, forward_vel), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)))
    else:
        ec.commands.base_velocity.forward_vel = forward_vel
        ec.commands.base_velocity.kx = 1.0
        ec.commands.base_velocity.ky = 1.0
        ec.commands.base_velocity.kyaw = 1.0
    return ec

def verify(env, mass):
    import numpy as np
    robot = env.scene['robot']
    ids = list(env.event_manager.get_term_cfg('add_base_mass').params['asset_cfg'].body_ids)
    assert len(ids) == 1 and robot.body_names[ids[0]] == 'base'
    actual = robot.root_physx_view.get_masses().cpu().numpy()
    default = robot.data.default_mass.cpu().numpy()
    np.testing.assert_allclose(actual[:, ids] - default[:, ids], mass, atol=1e-05)
    actuator = robot.actuators['base_legs']
    np.testing.assert_allclose(actuator.stiffness.cpu().numpy(), 25.0, atol=1e-05)
    np.testing.assert_allclose(actuator.damping.cpu().numpy(), 0.5, atol=1e-05)
    assert 'joint_stiffness_and_damping' not in env.event_manager._mode_term_names.get('reset', [])
    assert 'joint_friction' not in env.event_manager._mode_term_names.get('reset', [])
    mats = robot.root_physx_view.get_material_properties().cpu().numpy()
    from simdist.rl.go2_mdp import randomize_rigid_body_material_same
    event = next((t.func for t in env.event_manager._mode_term_cfgs['startup'] if isinstance(t.func, randomize_rigid_body_material_same)))
    foot_shapes = []
    for bid in event.asset_cfg.body_ids:
        start = sum(event.num_shapes_per_body[:bid])
        foot_shapes.extend(range(start, start + event.num_shapes_per_body[bid]))
    np.testing.assert_allclose(mats[:, foot_shapes, 0:2], 0.8, atol=1e-05)
    np.testing.assert_allclose(mats[:, foot_shapes, 2], 0.05, atol=1e-05)
    return {'base_mass_kg': float(actual[0, ids[0]]), 'default_base_mass_kg': float(default[0, ids[0]]), 'foot_shape_count': len(foot_shapes), 'kp': 25.0, 'kd': 0.5}
