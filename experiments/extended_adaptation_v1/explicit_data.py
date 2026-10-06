"""25-history/25-future windows split by rollout seed, never by trailing steps."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import h5py
import numpy as np
H = T = 25

class ExplicitSeedData:

    def __init__(self, train_paths, heldout_paths, latest_heldout_path, model_cfg):
        self.episodes = []
        self.train = []
        self.heldout = []
        self.latest = []
        self.sources = []
        self.proprio_noise = np.asarray([value for term in model_cfg['system']['proprio_obs']['types'] for value in [term['noise']] * term['dim']], dtype=np.float32)
        self.height_noise = next((float(term['noise']) for term in model_cfg['system']['extero_obs']['types'] if term['name'] == 'height_scan'))
        train_paths = list(train_paths)
        heldout_paths = list(heldout_paths)
        assert train_paths and heldout_paths and (latest_heldout_path in heldout_paths)
        assert not set(train_paths).intersection(heldout_paths)
        for partition, paths in (('train', train_paths), ('heldout', heldout_paths)):
            for path in paths:
                before = len(getattr(self, partition))
                with h5py.File(path, 'r') as file:
                    for name in sorted(file.keys()):
                        episode = {key: np.asarray(file[name][key], dtype=np.float32) for key in ('proprio_obs', 'extero_obs', 'actions', 'commands', 'reward')}
                        length = len(episode['actions'])
                        assert all((len(value) == length for value in episode.values()))
                        episode_id = len(self.episodes)
                        self.episodes.append(episode)
                        for start in range(max(0, length - H - T)):
                            ref = (episode_id, start)
                            getattr(self, partition).append(ref)
                            if path == latest_heldout_path:
                                self.latest.append(ref)
                self.sources.append({'path': str(path), 'partition': partition, 'windows': len(getattr(self, partition)) - before})
        assert self.train and self.heldout and self.latest

    def batch(self, refs, augment=False):
        items = []
        for episode_id, start in refs:
            e = self.episodes[episode_id]
            t, u = (start + H, start + H + T)
            items.append({'model_in': {'proprio_obs_hist': e['proprio_obs'][start:t + 1], 'extero_obs': e['extero_obs'][t], 'acts_hist': e['actions'][start:t], 'fut_acts': e['actions'][t:u], 'fut_cmds': e['commands'][t:u + 1]}, 'labels': {'proprio_obs': e['proprio_obs'][t + 1:u + 1], 'extero_obs': e['extero_obs'][t + 1:u + 1], 'rewards': e['reward'][t:u], 'values': np.zeros(T, np.float32), 'actions': e['actions'][t:u], 'exp_pol_flags': np.zeros(T, np.bool_)}})
        result = {part: {key: np.stack([item[part][key] for item in items]) for key in items[0][part]} for part in ('model_in', 'labels')}
        if augment:
            x = result['model_in']
            x['proprio_obs_hist'] = x['proprio_obs_hist'] + np.random.randn(*x['proprio_obs_hist'].shape) * self.proprio_noise
            x['extero_obs'] = x['extero_obs'] + np.random.randn(*x['extero_obs'].shape) * self.height_noise
        return result
