"""Scratch-only same-env state capture, based on experiments/check_s2.py."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
from copy import deepcopy
import numpy as np
import torch

def direct_tensors(obj):
    return {key: value.clone() for key, value in vars(obj).items() if isinstance(value, torch.Tensor)}

def put_tensors(obj, values):
    for key, saved in values.items():
        old = getattr(obj, key, None)
        if isinstance(old, torch.Tensor) and old.shape == saved.shape:
            old.copy_(saved)
        else:
            setattr(obj, key, saved.clone())

def capture_env(env):
    robot = env.scene['robot']
    command = env.command_manager.get_term('base_velocity')
    return {'scene': env.scene.get_state(is_relative=True), 'root_pos_audit': robot.data.root_pos_w.clone(), 'joint_pos_audit': robot.data.joint_pos.clone(), 'env': {key: getattr(env, key).clone() for key in ('episode_length_buf', 'reset_buf', 'reset_terminated', 'reset_time_outs', 'reward_buf')}, 'counters': {key: getattr(env, key) for key in ('common_step_counter', '_sim_step_counter')}, 'command': direct_tensors(command), 'action': direct_tensors(env.action_manager), 'action_terms': {key: direct_tensors(term) for key, term in env.action_manager._terms.items()}, 'robot_data': direct_tensors(robot.data), 'robot_private': {key: value.clone() for key, value in vars(robot).items() if key.startswith('_joint_') and isinstance(value, torch.Tensor)}, 'actuator': {key: direct_tensors(item) for key, item in robot.actuators.items()}, 'sensors': {key: {'base': direct_tensors(item), 'data': direct_tensors(item.data)} for key, item in env.scene.sensors.items()}, 'history': {group: {term: {'tensors': direct_tensors(buffer), 'pointer': buffer._pointer} for term, buffer in terms.items()} for group, terms in env.observation_manager._group_obs_term_history_buffer.items()}, 'reward_sums': {key: value.clone() for key, value in env.reward_manager._episode_sums.items()}, 'cpu_rng': torch.get_rng_state().clone(), 'cuda_rng': [value.clone() for value in torch.cuda.get_rng_state_all()]}

def restore_env(env, saved):
    env.scene.reset_to(saved['scene'], env_ids=torch.arange(env.num_envs, device=env.device), is_relative=True)
    for key, value in saved['env'].items():
        getattr(env, key).copy_(value)
    for key, value in saved['counters'].items():
        setattr(env, key, value)
    put_tensors(env.command_manager.get_term('base_velocity'), saved['command'])
    put_tensors(env.action_manager, saved['action'])
    for key, values in saved['action_terms'].items():
        put_tensors(env.action_manager._terms[key], values)
    robot = env.scene['robot']
    put_tensors(robot.data, saved['robot_data'])
    put_tensors(robot, saved['robot_private'])
    for key, values in saved['actuator'].items():
        put_tensors(robot.actuators[key], values)
    for key, values in saved['sensors'].items():
        put_tensors(env.scene.sensors[key], values['base'])
        put_tensors(env.scene.sensors[key].data, values['data'])
    for group, terms in saved['history'].items():
        for term, values in terms.items():
            buffer = env.observation_manager._group_obs_term_history_buffer[group][term]
            put_tensors(buffer, values['tensors'])
            buffer._pointer = values['pointer']
    for key, value in saved['reward_sums'].items():
        env.reward_manager._episode_sums[key].copy_(value)
    torch.set_rng_state(saved['cpu_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])

def capture_controller(controller, previous, obs, observation_history, command_history):
    ring = {}
    for key, buffer in zip(controller.buf.keys, controller.buf.buffers):
        ring[key] = (buffer.buffer.copy(), buffer.idx, buffer.full)
    return {'ring': ring, 'episode_starts': (controller.buf.episode_starts.buffer.copy(), controller.buf.episode_starts.idx, controller.buf.episode_starts.full), 'mppi_state': controller.mppi_state, 'fut_cmds': controller.fut_cmds.copy(), 'previous': previous.copy(), 'obs': deepcopy(obs), 'observation_history': deepcopy(observation_history), 'command_history': deepcopy(command_history)}

def restore_controller(controller, saved):
    for key, buffer in zip(controller.buf.keys, controller.buf.buffers):
        array, index, full = saved['ring'][key]
        np.copyto(buffer.buffer, array)
        buffer.idx, buffer.full = (index, full)
    array, index, full = saved['episode_starts']
    np.copyto(controller.buf.episode_starts.buffer, array)
    controller.buf.episode_starts.idx, controller.buf.episode_starts.full = (index, full)
    controller.mppi_state = saved['mppi_state']
    controller.fut_cmds = saved['fut_cmds'].copy()
