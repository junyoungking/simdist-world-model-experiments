"""Recovery helpers; frozen selection rule and temporary rollout-buffer offload."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import numpy as np
from scipy.stats import t

def select(history, best, candidate):
    item = history[-1]
    completed = item['target_iteration']
    if best is None or (item['mean_Return'], -item['ET_events'], -completed) > (best['mean_Return'], -best['ET_events'], -best['target_iteration']):
        best = item
    last = history[-4:]
    stable = len(last) == 4 and np.ptp([x['mean_Return'] for x in last]) <= 1.0 and (np.ptp([x['ET_seed_rate'] for x in last]) <= 0.100000001) and (np.ptp([x['mean_vx_RMSE'] for x in last]) <= 0.02)
    if candidate is None and stable:
        candidate = {'iteration': completed, 'stats': item}
    stop = False
    confirmation = None
    if candidate is not None and completed >= candidate['iteration'] + 1000:
        delta = np.array(item['per_seed_Return']) - np.array(candidate['stats']['per_seed_Return'])
        half = t.ppf(0.975, 9) * delta.std(ddof=1) / np.sqrt(10)
        confirmation = {'candidate_iteration': candidate['iteration'], 'confirmation_iteration': completed, 'paired_change_mean': float(delta.mean()), 'paired_change_SD': float(delta.std(ddof=1)), 'paired_change_t95': [float(delta.mean() - half), float(delta.mean() + half)], 'candidate_to_current_ET_rate_change': item['ET_seed_rate'] - candidate['stats']['ET_seed_rate'], 'candidate_to_current_vx_RMSE_change': item['mean_vx_RMSE'] - candidate['stats']['mean_vx_RMSE']}
        stop = stable and delta.mean() <= 1.0 and (delta.mean() + half <= 1.0) and (abs(confirmation['candidate_to_current_ET_rate_change']) <= 0.100000001) and (abs(confirmation['candidate_to_current_vx_RMSE_change']) <= 0.02)
        if not stop:
            candidate = None
    return (best, candidate, bool(stop), confirmation)

def offload_storage(storage):
    import torch
    assert storage.step == 0, 'Offload only after completed PPO update'
    assert storage.saved_hidden_states_a is None and storage.saved_hidden_states_c is None
    devices = {}
    for name in vars(storage):
        value = getattr(storage, name)
        if torch.is_tensor(value) and value.device.type == 'cuda':
            devices[name] = value.device
            setattr(storage, name, value.to('cpu'))
    return devices

def restore_storage(storage, devices):
    for name, device in devices.items():
        setattr(storage, name, getattr(storage, name).to(device))
