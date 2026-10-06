"""Fixed-parameter gap configurations and physical readback for scratch runs."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import numpy as np
import sys
from pathlib import Path
from go2_probe_common import config
GAPS = ('friction015', 'gain070', 'mass8')

def make_gap(seed, gap):
    assert gap in GAPS
    cfg = config(seed, 8.0 if gap == 'mass8' else 0.0, 1.0)
    if gap == 'friction015':
        material = cfg.events.physics_material.params
        material['static_friction_range'] = (0.15, 0.15)
        material['dynamic_friction_range'] = (0.15, 0.15)
    elif gap == 'gain070':
        motor = cfg.scene.robot.actuators['base_legs']
        cfg.scene.robot = cfg.scene.robot.replace(actuators={'base_legs': motor.replace(stiffness=17.5, damping=0.35)})
    return cfg

def verify_gap(env, gap):
    from simdist.rl.go2_mdp import randomize_rigid_body_material_same
    robot = env.scene['robot']
    body_ids = list(env.event_manager.get_term_cfg('add_base_mass').params['asset_cfg'].body_ids)
    assert len(body_ids) == 1 and robot.body_names[body_ids[0]] == 'base'
    actual = robot.root_physx_view.get_masses().cpu().numpy()
    default = robot.data.default_mass.cpu().numpy()
    expected_mass = 8.0 if gap == 'mass8' else 0.0
    np.testing.assert_allclose(actual[:, body_ids] - default[:, body_ids], expected_mass, atol=1e-05)
    actuator = robot.actuators['base_legs']
    kp, kd = (17.5, 0.35) if gap == 'gain070' else (25.0, 0.5)
    np.testing.assert_allclose(actuator.stiffness.cpu().numpy(), kp, atol=1e-05)
    np.testing.assert_allclose(actuator.damping.cpu().numpy(), kd, atol=1e-05)
    np.testing.assert_allclose(actuator.effort_limit.cpu().numpy(), 23.5, atol=1e-05)
    event = next((term.func for term in env.event_manager._mode_term_cfgs['startup'] if isinstance(term.func, randomize_rigid_body_material_same)))
    foot_shapes = []
    for body_id in event.asset_cfg.body_ids:
        start = sum(event.num_shapes_per_body[:body_id])
        foot_shapes.extend(range(start, start + event.num_shapes_per_body[body_id]))
    material = robot.root_physx_view.get_material_properties().cpu().numpy()
    friction = 0.15 if gap == 'friction015' else 0.8
    np.testing.assert_allclose(material[:, foot_shapes, :2], friction, atol=1e-05)
    np.testing.assert_allclose(material[:, foot_shapes, 2], 0.05, atol=1e-05)
    return {'base_mass_kg': float(actual[0, body_ids[0]]), 'mass_added_kg': expected_mass, 'foot_friction': friction, 'foot_restitution': 0.05, 'kp': kp, 'kd': kd, 'effort_limit': 23.5, 'foot_shape_count': len(foot_shapes)}
