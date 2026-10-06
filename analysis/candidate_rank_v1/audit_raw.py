"""Audit per-snapshot numeric arrays against candidate summaries."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import csv
import json
import math
import argparse
from pathlib import Path
import numpy as np
ROOT = artifact_dir('candidate_rank_v1') / 'full'
CONDITIONS = ('cp30000_mass8', 'm0_mass0', 'm0_mass8')
parser = argparse.ArgumentParser()
parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
args = parser.parse_args()

def main():
    total = 0
    branches = 0
    full25 = 0
    max_score_error = max_realized_error = max_head_error = max_plan_action_error = 0.0
    for condition in args.conditions:
        for seed in range(23601, 23609):
            seed_root = ROOT / condition / f'seed_{seed}'
            summary = json.loads((seed_root / 'summary.json').read_text())
            paths = sorted(seed_root.glob('snapshot_????'))
            assert len(paths) == summary['snapshots_total'] == summary['snapshots_processed']
            for path in paths:
                rows = list(csv.DictReader((path / 'candidates.csv').open()))
                assert len(rows) == 68
                source = path / 'sequences.npz'
                if condition == 'cp30000_mass8':
                    source = path / 'sequences_with_plans.npz'
                with np.load(source) as raw:
                    arrays = {key: raw[key] for key in raw.files}
                assert arrays['predicted_reward_sequence'].shape == (68, 25)
                assert arrays['actual_env_reward_sequence'].shape == (68, 200)
                assert arrays['planned_action_sequences'].shape == (68, 25, 12)
                assert arrays['executed_action_sequences'].shape == (68, 25, 12)
                assert arrays['assumed_future_commands'].shape == (26, 3)
                assert arrays['actual_latent_reward_head_sequence'].shape == (68, 25)
                assert arrays['actual_latent_value_head_sequence'].shape == (68, 25)
                for index, row in enumerate(rows):
                    steps = int(row['actual_steps'])
                    observed = arrays['actual_env_reward_sequence'][index]
                    assert np.isfinite(observed[:steps]).all() and np.isnan(observed[steps:]).all()
                    expected = sum((0.99 ** k * float(observed[k]) for k in range(min(25, steps))))
                    max_realized_error = max(max_realized_error, abs(expected - float(row['actual_reward25_discounted'])))
                    predicted = sum((0.99 ** k * float(v) for k, v in enumerate(arrays['predicted_reward_sequence'][index])))
                    max_score_error = max(max_score_error, abs(predicted - float(row['pred_reward_discounted'])))
                    if index < 66:
                        max_plan_action_error = max(max_plan_action_error, float(np.max(np.abs(arrays['planned_action_sequences'][index, :steps] - arrays['executed_action_sequences'][index, :steps]))))
                    valid = bool(int(row['actual_25_complete']))
                    assert valid == bool(arrays['actual_head_valid'][index])
                    if valid:
                        full25 += 1
                        head = sum((0.99 ** k * float(v) for k, v in enumerate(arrays['actual_latent_reward_head_sequence'][index])))
                        max_head_error = max(max_head_error, abs(head - float(row['actual_head_reward_discounted'])))
                        assert math.isclose(float(arrays['actual_latent_value_head_sequence'][index, -1]), float(row['actual_head_terminal_value_raw']), abs_tol=1e-06)
                    else:
                        assert row['actual_head_terminal_value_raw'] == ''
                    branches += 1
                total += 1
    result = {'snapshots': total, 'branches': branches, 'full25_branches': full25, 'max_pred_reward_reconstruction_error': max_score_error, 'max_actual_reward_reconstruction_error': max_realized_error, 'max_actual_head_reconstruction_error': max_head_error, 'max_open_loop_planned_vs_executed_prefix_error': max_plan_action_error}
    print('RAW_AUDIT', json.dumps(result), flush=True)
    output = ROOT.parent / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'raw_audit.json').write_text(json.dumps(result, indent=2) + '\n')
if __name__ == '__main__':
    main()
