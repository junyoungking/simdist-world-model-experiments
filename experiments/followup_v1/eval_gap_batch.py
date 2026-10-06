"""One v2 Go2 episode with default-joint reset and expert warm-up."""
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
import csv
import json
import logging
import sys
import time
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--controller', choices=('mppi', 'base', 'expert'), required=True)
parser.add_argument('--gap', choices=('source', 'mass8', 'mass5', 'friction010', 'friction015', 'gain070', 'effort16'), required=True)
parser.add_argument('--seeds', type=int, nargs='+', required=True)
parser.add_argument('--forward-vel', type=float, choices=(0.5, 1.0), required=True)
parser.add_argument('--warmup-only', action='store_true')
parser.add_argument('--out-root', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, default=None)
parser.add_argument('--checkpoint-step', type=int, default=None)
args = parser.parse_args()
app = AppLauncher(headless=True).app
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from isaaclab.envs import ManagerBasedRLEnv
from simdist.utils import model as model_utils, config as config_utils
from simdist.control.mppi import MppiController
from go2_probe_common import ROOT, MODEL, EXPERT, config
from flax import nnx
import omni.usd
logging.getLogger('jax').setLevel(logging.WARNING)
cache = {}

def gap_config(seed):
    mass = {'mass8': 8.0, 'mass5': 5.0}.get(args.gap, 0.0)
    cfg = config(seed, mass, args.forward_vel)
    if args.gap in ('friction010', 'friction015'):
        friction = 0.1 if args.gap == 'friction010' else 0.15
        material = cfg.events.physics_material.params
        material['static_friction_range'] = (friction, friction)
        material['dynamic_friction_range'] = (friction, friction)
    elif args.gap == 'gain070':
        motor = cfg.scene.robot.actuators['base_legs']
        cfg.scene.robot = cfg.scene.robot.replace(actuators={'base_legs': motor.replace(stiffness=17.5, damping=0.35)})
    elif args.gap == 'effort16':
        motor = cfg.scene.robot.actuators['base_legs']
        cfg.scene.robot = cfg.scene.robot.replace(actuators={'base_legs': motor.replace(effort_limit=16.0, saturation_effort=16.0)})
    return (cfg, mass)

def verify_gap(env, mass):
    import numpy as np
    from simdist.rl.go2_mdp import randomize_rigid_body_material_same
    robot = env.scene['robot']
    ids = list(env.event_manager.get_term_cfg('add_base_mass').params['asset_cfg'].body_ids)
    assert len(ids) == 1 and robot.body_names[ids[0]] == 'base'
    actual = robot.root_physx_view.get_masses().cpu().numpy()
    default = robot.data.default_mass.cpu().numpy()
    np.testing.assert_allclose(actual[:, ids] - default[:, ids], mass, atol=1e-05)
    actuator = robot.actuators['base_legs']
    kp = 17.5 if args.gap == 'gain070' else 25.0
    kd = 0.35 if args.gap == 'gain070' else 0.5
    np.testing.assert_allclose(actuator.stiffness.cpu().numpy(), kp, atol=1e-05)
    np.testing.assert_allclose(actuator.damping.cpu().numpy(), kd, atol=1e-05)
    effort = 16.0 if args.gap == 'effort16' else 23.5
    np.testing.assert_allclose(actuator.effort_limit.cpu().numpy(), effort, atol=1e-05)
    assert abs(float(actuator.cfg.saturation_effort) - effort) < 1e-05
    assert 'joint_stiffness_and_damping' not in env.event_manager._mode_term_names.get('reset', [])
    assert 'joint_friction' not in env.event_manager._mode_term_names.get('reset', [])
    materials = robot.root_physx_view.get_material_properties().cpu().numpy()
    event = next((t.func for t in env.event_manager._mode_term_cfgs['startup'] if isinstance(t.func, randomize_rigid_body_material_same)))
    foot_shapes = []
    for body_id in event.asset_cfg.body_ids:
        start = sum(event.num_shapes_per_body[:body_id])
        foot_shapes.extend(range(start, start + event.num_shapes_per_body[body_id]))
    friction = {'friction010': 0.1, 'friction015': 0.15}.get(args.gap, 0.8)
    np.testing.assert_allclose(materials[:, foot_shapes, 0:2], friction, atol=1e-05)
    np.testing.assert_allclose(materials[:, foot_shapes, 2], 0.05, atol=1e-05)
    return {'mass_added_kg': mass, 'base_mass_kg': float(actual[0, ids[0]]), 'foot_friction': friction, 'foot_restitution': 0.05, 'kp': kp, 'kd': kd, 'effort_limit': effort, 'saturation_effort': float(actuator.cfg.saturation_effort), 'foot_shape_count': len(foot_shapes)}

class BaseOnlyController(MppiController):
    """Compile only the base head, retaining the original history/input path."""

    def _init_fn(self):
        self._get_base_policy_actions(self._make_model_inputs(self.dummy_actions))

def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    if not args.warmup_only and args.controller in ('mppi', 'base') and ('controller' not in cache):
        with initialize_config_dir(config_dir=str(simdist_root() / 'config'), version_base=None):
            cfg = OmegaConf.to_container(compose(config_name='simulate_go2'), resolve=True)
        cfg['control']['seed'] = args.seed
        checkpoint = args.checkpoint.resolve() if args.checkpoint else MODEL
        requested_step = args.checkpoint_step if args.checkpoint else 367004
        model, model_cfg, model_step = model_utils.load_model_from_ckpt(str(checkpoint), requested_step)
        names = config_utils.proprio_obs_names_from_sys_config(model_cfg['system'])
        cls = MppiController if args.controller == 'mppi' else BaseOnlyController
        controller = cls(model, model_cfg, cfg['control'])
        cache.update(controller=controller, names=names, model_step=model_step)
    elif not args.warmup_only and args.controller in ('mppi', 'base'):
        controller, names, model_step = (cache['controller'], cache['names'], cache['model_step'])
        controller.rngs = nnx.Rngs(args.seed)
        controller.rngs()
    else:
        controller = None
        names = None
        model_step = None
    if 'expert' not in cache:
        cache['expert'] = torch.jit.load(str(EXPERT), map_location='cuda:0').eval()
    expert = cache['expert']
    ec, mass = gap_config(args.seed)
    ec.episode_length_s = 20.5
    env = ManagerBasedRLEnv(ec)
    try:
        obs, _ = env.reset()
        robot = env.scene['robot']
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
        env.sim.forward()
        env.scene.update(env.physics_dt)
        obs = env.observation_manager.compute()
        physical = verify_gap(env, mass)
        assert obs['policy'].shape == (1, 433)
        joint_reset_error = float(torch.max(torch.abs(robot.data.joint_pos - robot.data.default_joint_pos)).item())
        joint_reset_vel_error = float(torch.max(torch.abs(robot.data.joint_vel)).item())
        assert joint_reset_error < 1e-05 and joint_reset_vel_error < 1e-05
        zero = np.zeros(12, np.float32)
        previous = zero.copy()

        def model_input(ob, prev):
            prop = np.concatenate([ob['obs'][n][0].detach().cpu().numpy() for n in names]).astype(np.float32)
            extero = ob['obs']['height_scan'][0].detach().cpu().numpy().astype(np.float32)
            assert prop.shape == (35,) and extero.shape == (315,)
            return {'proprio_obs': prop, 'extero_obs': extero, 'prev_action': prev}
        if controller is not None:
            initial_cmd = env.command_manager.get_command('base_velocity')[0].detach().cpu().numpy().copy()
            if not controller.initialized:
                controller.initialize(model_input(obs, zero), initial_cmd)
            controller.reset(model_input(obs, zero), initial_cmd)
        warmup = []
        warmup_failed = False
        for i in range(25):
            cmd = env.command_manager.get_command('base_velocity')[0].detach().cpu().numpy().copy()
            if controller is not None:
                controller.set_fut_cmd(cmd)
                controller.update(model_input(obs, previous))
            with torch.inference_mode():
                action = expert(obs['policy']).squeeze(0).detach().cpu().numpy().astype(np.float32)
            assert np.isfinite(action).all()
            obs, reward, terminated, truncated, _ = env.step(torch.from_numpy(action[None]).to(env.device))
            reasons = [n for n in env.termination_manager.active_terms if bool(env.termination_manager.get_term(n)[0])]
            warmup.append({'step': i + 1, 'reward': float(reward[0]), 'terminated': int(bool(terminated[0])), 'truncated': int(bool(truncated[0])), 'termination_terms': '|'.join(reasons)})
            previous = action
            if bool(terminated[0]) or bool(truncated[0]):
                warmup_failed = True
                break
        write_csv(args.out / 'warmup_steps.csv', warmup, list(warmup[0]))
        print(f'WARMUP_DONE steps={len(warmup)} failed={warmup_failed}', flush=True)
        steps = []
        if not warmup_failed and (not args.warmup_only):
            for i in range(1000):
                if i == 0:
                    print('EVALUATION_BEGIN', flush=True)
                cmd = env.command_manager.get_command('base_velocity')[0].detach().cpu().numpy().copy()
                vx = float(robot.data.root_lin_vel_b[0, 0].item())
                if controller is not None:
                    controller.set_fut_cmd(cmd)
                    controller.update(model_input(obs, previous))
                    if args.controller == 'mppi':
                        action = np.asarray(controller.run_control()['actions'][0], np.float32)
                    else:
                        inputs = controller._make_model_inputs(controller.dummy_actions)
                        action = np.asarray(controller._get_base_policy_actions(inputs)[0], np.float32)
                else:
                    with torch.inference_mode():
                        action = expert(obs['policy']).squeeze(0).detach().cpu().numpy().astype(np.float32)
                assert action.shape == (12,) and np.isfinite(action).all()
                obs, reward, terminated, truncated, _ = env.step(torch.from_numpy(action[None]).to(env.device))
                reasons = [n for n in env.termination_manager.active_terms if bool(env.termination_manager.get_term(n)[0])]
                steps.append({'step': i + 1, 'reward': float(reward[0]), 'vx_body_pre': vx, 'command_vx': float(cmd[0]), 'tracking_error_abs': abs(vx - float(cmd[0])), 'terminated': int(bool(terminated[0])), 'truncated': int(bool(truncated[0])), 'termination_terms': '|'.join(reasons)})
                previous = action
                if bool(terminated[0]) or bool(truncated[0]):
                    break
                if (i + 1) % 100 == 0:
                    print(f'PROGRESS {args.controller} gap={args.gap} vel={args.forward_vel} seed={args.seed} {i + 1}/1000', flush=True)
        if steps:
            write_csv(args.out / 'evaluation_steps.csv', steps, list(steps[0]))
        outcome = 'warmup_failed' if warmup_failed else 'warmup_passed' if args.warmup_only else 'evaluated'
        summary = {'status': outcome, 'controller': args.controller, 'gap': args.gap, 'mass_added_kg': mass, 'forward_target_mps': args.forward_vel, 'seed': args.seed, 'model_step': model_step, 'physical_readback': physical, 'joint_reset_max_error': joint_reset_error, 'joint_velocity_reset_max_error': joint_reset_vel_error, 'warmup_steps': len(warmup), 'warmup_termination_count': sum((x['terminated'] for x in warmup)), 'warmup_termination_terms': [x['termination_terms'] for x in warmup if x['terminated'] or x['truncated']], 'episode_length': len(steps), 'fall': int(bool(steps and steps[-1]['terminated'])), 'end_reason': 'warmup_failed' if warmup_failed else 'warmup_only' if args.warmup_only else 'terminated' if steps and steps[-1]['terminated'] else 'timeout' if steps and steps[-1]['truncated'] else 'budget_censored', 'termination_terms': steps[-1]['termination_terms'] if steps else '', 'return': sum((x['reward'] for x in steps)), 'reward_per_step': float(np.mean([x['reward'] for x in steps])) if steps else None, 'mean_vx_body_mps': float(np.mean([x['vx_body_pre'] for x in steps])) if steps else None, 'mean_vx_abs_tracking_error_mps': float(np.mean([x['tracking_error_abs'] for x in steps])) if steps else None, 'duration_seconds': time.monotonic() - started}
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('SUMMARY', json.dumps(summary), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.close()
        omni.usd.get_context().new_stage()
if __name__ == '__main__':
    try:
        for seed in args.seeds:
            args.seed = seed
            suffix = 'v05' if args.forward_vel == 0.5 else 'v10'
            name = f'{args.controller}_{args.gap}_{suffix}_seed{seed}'
            args.out = args.out_root / name
            if (args.out / 'summary.json').exists():
                print('DONE', name, flush=True)
                continue
            print('START', name, flush=True)
            main()
            status = json.loads((args.out / 'summary.json').read_text())['status']
            if status not in ('evaluated', 'warmup_failed'):
                raise RuntimeError(f'{name} status={status}')
            print('FINISH', name, flush=True)
    finally:
        app.close()
