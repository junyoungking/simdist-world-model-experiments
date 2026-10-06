"""Released raw->processed path, with explicit stationary target eligibility."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import argparse
import json
from common import ROOT, config, route_paths, write_json, hashes

def process(name):
    import numpy as np
    import h5py
    from simdist.data.episode_aggregator import EpisodeAggregator
    from simdist.data.data_processor import DataProcessor
    from simdist.utils import paths
    route_paths()
    folder = ROOT / 'datasets' / name
    assert not (folder / 'dataset_manifest.json').exists(), 'Frozen dataset must not be reprocessed'
    rollout = json.loads((folder / 'rollout.json').read_text())
    assert hashes(folder / 'raw') == rollout['raw_hashes'], 'Raw file mutation'
    cfg = config('aggregate_realworld_data')
    cfg['dataset_name'] = name
    EpisodeAggregator(cfg).process_all_episodes()
    cfg = config('process_data')
    cfg['dataset_name'] = name
    DataProcessor(cfg).run()
    processed = paths.get_processed_data_dir(name, 'go2', 25, 25)
    with h5py.File(folder / 'raw_data.hdf5', 'r') as f:
        episode_lengths = [len(f['data'][k]['actions']) for k in f['data']]
    offsets = []
    offset = 0
    for length in episode_lengths:
        if length >= 60:
            offsets.append((offset, length))
            offset += length
    with h5py.File(paths.get_start_idxs_path(processed), 'r+') as f:
        starts = f['data'][:]
        keep = np.zeros(len(starts), dtype=bool)
        for offset, length in offsets:
            keep |= (starts >= offset + 25) & (starts + 50 < offset + length - 5)
        retained = starts[keep]
        f['data'].resize((len(retained),))
        f['data'][:] = retained
    assert len(retained), 'No eligible stationary windows'
    manifest = {'name': name, 'role': rollout['role'], 'seed': rollout.get('seed'), 'seeds': rollout.get('seeds', [rollout.get('seed')]), 'intervention_family': rollout.get('intervention_family'), 'friction': rollout['friction'], 'added_mass_kg': rollout.get('added_mass_kg', 0.0), 'kp': rollout.get('kp'), 'kd': rollout.get('kd'), 'fixed_kp': rollout.get('fixed_kp'), 'fixed_kd': rollout.get('fixed_kd'), 'raw_trajectory_count': len(rollout['episodes']), 'raw_rows': rollout['completed_steps'], 'raw_transition_count': rollout['raw_transition_count'], 'nominal_windows': rollout['nominal_windows'], 'released_processed_windows': len(starts), 'eligible_windows': len(retained), 'retained_fraction': len(retained) / rollout['nominal_windows'], 'stationary_filter': 'raw current row >=25; history may include real stabilization; no cross-episode windows', 'raw_hashes': rollout['raw_hashes'], 'processed_dir': processed, 'processed_hashes': hashes(processed)}
    write_json(folder / 'dataset_manifest.json', manifest)
    print(json.dumps(manifest), flush=True)
    return manifest
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    process(p.parse_args().name)
