"""Scratch-only plane/boxes configuration; retain the original observation and reward contract."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import sys
from pathlib import Path
import numpy as np
from go2_probe_common import config

def make_condition(seed, gap, terrain, friction, velocity):
    assert gap in ('source', 'mass8')
    cfg = config(seed, 8.0 if gap == 'mass8' else 0.0, velocity)
    if terrain == 'plane':
        cfg.scene.terrain.terrain_type = 'plane'
        cfg.scene.terrain.terrain_generator = None
        cfg.curriculum.terrain_levels = None
    cfg.events.physics_material.params['static_friction_range'] = (friction, friction)
    cfg.events.physics_material.params['dynamic_friction_range'] = (friction, friction)
    return cfg

def verify_condition(env, gap, terrain, friction):
    from simdist.rl.go2_mdp import randomize_rigid_body_material_same
    robot = env.scene['robot']
    ids = list(env.event_manager.get_term_cfg('add_base_mass').params['asset_cfg'].body_ids)
    assert len(ids) == 1 and robot.body_names[ids[0]] == 'base'
    mass = 8.0 if gap == 'mass8' else 0.0
    actual = robot.root_physx_view.get_masses().cpu().numpy()
    default = robot.data.default_mass.cpu().numpy()
    np.testing.assert_allclose(actual[:, ids] - default[:, ids], mass, atol=1e-05)
    actuator = robot.actuators['base_legs']
    for name, expected in (('stiffness', 25.0), ('damping', 0.5), ('effort_limit', 23.5)):
        np.testing.assert_allclose(getattr(actuator, name).cpu().numpy(), expected, atol=1e-05)
    assert actuator.cfg.saturation_effort == 23.5
    assert 'joint_stiffness_and_damping' not in env.event_manager._mode_term_names.get('reset', [])
    assert 'joint_friction' not in env.event_manager._mode_term_names.get('reset', [])
    event = next((t.func for t in env.event_manager._mode_term_cfgs['startup'] if isinstance(t.func, randomize_rigid_body_material_same)))
    shapes = []
    for body_id in event.asset_cfg.body_ids:
        start = sum(event.num_shapes_per_body[:body_id])
        shapes.extend(range(start, start + event.num_shapes_per_body[body_id]))
    materials = robot.root_physx_view.get_material_properties().cpu().numpy()
    np.testing.assert_allclose(materials[:, shapes, :2], friction, atol=1e-05)
    np.testing.assert_allclose(materials[:, shapes, 2], 0.05, atol=1e-05)
    assert env.cfg.scene.terrain.terrain_type == ('plane' if terrain == 'plane' else 'generator')
    hits = env.scene['height_scanner'].data.ray_hits_w[..., 2].cpu().numpy()
    span = float(np.max(hits) - np.min(hits))
    if terrain == 'plane':
        assert np.isfinite(hits).all() and span < 1e-05
        assert env.cfg.scene.terrain.terrain_generator is None
    return {'terrain': terrain, 'terrain_type': env.cfg.scene.terrain.terrain_type, 'height_scan_world_z_span': span, 'mass_added_kg': mass, 'base_mass_kg': float(actual[0, ids[0]]), 'foot_friction': friction, 'foot_restitution': 0.05, 'kp': 25.0, 'kd': 0.5, 'effort_limit': 23.5, 'foot_shape_count': len(shapes)}
