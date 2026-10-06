"""Leakage-separated 25/25 SimDist windows from v2 control-only rollouts."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import h5py
import numpy as np
H = 25
T = 25

class CurveData:

    def __init__(self, rollout_paths, model_cfg):
        self.episodes = []
        self.train = []
        self.heldout = []
        self.latest = []
        self.intervals = []
        self.proprio_noise = np.asarray([value for term in model_cfg['system']['proprio_obs']['types'] for value in [term['noise']] * term['dim']], dtype=np.float32)
        self.height_noise = next((float(term['noise']) for term in model_cfg['system']['extero_obs']['types'] if term['name'] == 'height_scan'))
        for interval_index, path in enumerate(rollout_paths):
            with h5py.File(path, 'r') as file:
                names = sorted(file.keys())
                budget = sum((len(file[name]['actions']) for name in names))
                split = budget - int(round(0.1 * budget))
                offset = 0
                train_start, held_start, embargo = (len(self.train), len(self.heldout), 0)
                for name in names:
                    group = file[name]
                    episode = {key: np.asarray(group[key], dtype=np.float32) for key in ('proprio_obs', 'extero_obs', 'actions', 'commands', 'reward')}
                    length = len(episode['actions'])
                    assert all((len(v) == length for v in episode.values()))
                    ep_id = len(self.episodes)
                    self.episodes.append(episode)
                    for local_start in range(max(0, length - H - T)):
                        global_start = offset + local_start
                        if global_start + H + T < split:
                            self.train.append((ep_id, local_start))
                        elif global_start >= split:
                            self.heldout.append((ep_id, local_start))
                            if interval_index == len(rollout_paths) - 1:
                                self.latest.append((ep_id, local_start))
                        else:
                            embargo += 1
                    offset += length
                assert offset == budget
                self.intervals.append({'interval': interval_index + 1, 'control_steps': budget, 'heldout_control_steps': budget - split, 'train_windows': len(self.train) - train_start, 'heldout_windows': len(self.heldout) - held_start, 'embargo_windows': embargo, 'episodes': len(names)})
        assert self.train and self.heldout and self.latest, self.intervals

    def batch(self, refs, augment=False):
        items = []
        for episode_id, start in refs:
            e = self.episodes[episode_id]
            t = start + H
            u = t + T
            items.append({'model_in': {'proprio_obs_hist': e['proprio_obs'][start:t + 1], 'extero_obs': e['extero_obs'][t], 'acts_hist': e['actions'][start:t], 'fut_acts': e['actions'][t:u], 'fut_cmds': e['commands'][t:u + 1]}, 'labels': {'proprio_obs': e['proprio_obs'][t + 1:u + 1], 'extero_obs': e['extero_obs'][t + 1:u + 1], 'rewards': e['reward'][t:u], 'values': np.zeros(T, np.float32), 'actions': e['actions'][t:u], 'exp_pol_flags': np.zeros(T, np.bool_)}})
        result = {part: {key: np.stack([item[part][key] for item in items]) for key in items[0][part]} for part in ('model_in', 'labels')}
        if augment:
            x = result['model_in']
            x['proprio_obs_hist'] = x['proprio_obs_hist'] + np.random.randn(*x['proprio_obs_hist'].shape) * self.proprio_noise
            x['extero_obs'] = x['extero_obs'] + np.random.randn(*x['extero_obs'].shape) * self.height_noise
        return result
