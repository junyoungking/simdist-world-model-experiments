"""Read-only reanalysis of saved candidate scores, actions, rewards and commands.

This script does not import Isaac Lab, load a model or start a simulator. It reads
the saved candidate-level experiment and writes derived CSV files under scratch.
"""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import csv
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = artifact_dir('candidate_rank_v1')
SOURCE = ROOT / 'full'
OUT = ROOT / 'decomposition_v1'
CONDITIONS = ('cp30000_mass8', 'm0_mass0', 'm0_mass8')
KINDS = ('mean_plan', 'last_iteration', 'noise_0.1', 'noise_0.3', 'noise_1.0', 'base_proposal', 'expert_closed', 'base_closed')
GROUPS = ('all', 'near_fall_100', 'other')
DISCOUNT = 0.99
WEIGHTS = DISCOUNT ** np.arange(25, dtype=np.float64)
FINAL_WEIGHT = DISCOUNT ** 26
NARROW_INDICES = np.asarray(list(range(33)) + [65], dtype=int)

def write_csv(name, rows):
    path = OUT / name
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path

def stats(values):
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return (len(x), float(np.mean(x)) if len(x) else math.nan, float(np.std(x, ddof=1)) if len(x) > 1 else math.nan)

def rho(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(spearmanr(x, y).statistic)

def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])

def score_rank(a, b):
    return int(1 + np.sum(np.asarray(a) > float(b)))

def rank_pair(a, b):
    a, b = (np.asarray(a), np.asarray(b))
    return (rho(a, b), int(len(set(np.argsort(-a)[:8]) & set(np.argsort(-b)[:8]))))

def group_rows(rows, group):
    if group == 'all':
        return rows
    return [r for r in rows if ('near_fall_100' if r['near_fall_100'] else 'other') == group]

def add_stats(out, label, vals):
    n, mean, sd = stats(vals)
    out[f'n_{label}'] = n
    out[f'{label}_mean'] = mean
    out[f'{label}_sample_sd'] = sd

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    branches = []
    steps = []
    horizon_values = defaultdict(list)
    ranks = []
    expert_ranks = []
    action_cells = defaultdict(list)
    audit = {'snapshots': 0, 'complete_branches': 0, 'max_pred_reward_vs_csv': 0.0, 'max_head_reward_vs_csv': 0.0, 'max_env_reward_vs_csv': 0.0, 'max_decomposition_identity': 0.0, 'max_command_mae_vs_csv': 0.0, 'max_value_identity': 0.0, 'max_observed_value_vs_csv': 0.0, 'max_predicted_total_vs_csv': 0.0}
    for condition in CONDITIONS:
        snapshot_dirs = sorted((SOURCE / condition).glob('seed_*/snapshot_????'))
        assert snapshot_dirs, condition
        for snapshot_dir in snapshot_dirs:
            seed = int(snapshot_dir.parent.name.split('_')[-1])
            step = int(snapshot_dir.name.split('_')[-1])
            candidates = list(csv.DictReader((snapshot_dir / 'candidates.csv').open()))
            assert len(candidates) == 68
            sequence_file = snapshot_dir / 'sequences_with_plans.npz' if condition == 'cp30000_mass8' else snapshot_dir / 'sequences.npz'
            with np.load(sequence_file) as data:
                predicted = np.asarray(data['predicted_reward_sequence'], np.float64)
                observed_head = np.asarray(data['actual_latent_reward_head_sequence'], np.float64)
                observed_value = np.asarray(data['actual_latent_value_head_sequence'], np.float64)
                environment = np.asarray(data['actual_env_reward_sequence'][:, :25], np.float64)
                complete = np.asarray(data['actual_head_valid'], bool)
                planned = np.asarray(data['planned_action_sequences'], np.float64)
                executed = np.asarray(data['executed_action_sequences'], np.float64)
                assumed_command = np.asarray(data['assumed_future_commands'][:25], np.float64)
                actual_command = np.asarray(data['actual_command_sequence'][:, :25], np.float64)
            assert predicted.shape == observed_head.shape == (68, 25)
            assert observed_value.shape == (68, 25)
            assert environment.shape == (68, 25)
            assert planned.shape == executed.shape == (68, 25, 12)
            assert assumed_command.shape == (25, 3) and actual_command.shape == (68, 25, 3)
            assert complete.tolist() == [bool(int(r['actual_25_complete'])) for r in candidates]
            near = bool(int(candidates[0]['near_main_fall_100']))
            audit['snapshots'] += 1
            pred_sum = np.sum(predicted * WEIGHTS[None, :], axis=1)
            head_sum = np.sum(observed_head * WEIGHTS[None, :], axis=1)
            env_sum = np.sum(environment * WEIGHTS[None, :], axis=1)
            dynamic_sum = pred_sum - head_sum
            head_error_sum = head_sum - env_sum
            total_bias = pred_sum - env_sum
            command_mae = np.mean(np.abs(actual_command - assumed_command[None, :, :]), axis=(1, 2))
            action_rms = np.sqrt(np.mean((executed - planned[0][None]) ** 2, axis=(1, 2)))
            value_pred = np.asarray([float(r['pred_terminal_value_raw']) for r in candidates])
            value_actual = observed_value[:, -1]
            critic = np.asarray([float(r['expert_critic_step25']) if r['expert_critic_step25'] else math.nan for r in candidates])
            for idx, row in enumerate(candidates):
                if not complete[idx]:
                    continue
                audit['complete_branches'] += 1
                assert np.isfinite(environment[idx]).all()
                assert np.isfinite(actual_command[idx]).all()
                assert np.isfinite(critic[idx])
                audit['max_pred_reward_vs_csv'] = max(audit['max_pred_reward_vs_csv'], abs(pred_sum[idx] - float(row['pred_reward_discounted'])))
                audit['max_head_reward_vs_csv'] = max(audit['max_head_reward_vs_csv'], abs(head_sum[idx] - float(row['actual_head_reward_discounted'])))
                audit['max_env_reward_vs_csv'] = max(audit['max_env_reward_vs_csv'], abs(env_sum[idx] - float(row['actual_reward25_discounted'])))
                audit['max_decomposition_identity'] = max(audit['max_decomposition_identity'], abs(dynamic_sum[idx] + head_error_sum[idx] - total_bias[idx]))
                audit['max_command_mae_vs_csv'] = max(audit['max_command_mae_vs_csv'], abs(command_mae[idx] - float(row['command_mae'])))
                audit['max_value_identity'] = max(audit['max_value_identity'], abs(value_pred[idx] - value_actual[idx] + (value_actual[idx] - critic[idx]) - (value_pred[idx] - critic[idx])))
                audit['max_observed_value_vs_csv'] = max(audit['max_observed_value_vs_csv'], abs(value_actual[idx] - float(row['actual_head_terminal_value_raw'])))
                audit['max_predicted_total_vs_csv'] = max(audit['max_predicted_total_vs_csv'], abs(pred_sum[idx] + FINAL_WEIGHT * value_pred[idx] - float(row['pred_total'])))
                branch = {'condition': condition, 'seed': seed, 'snapshot_step': step, 'near_fall_100': int(near), 'candidate_index': idx, 'kind': row['candidate_kind'], 'pred_reward25': pred_sum[idx], 'observed_head_reward25': head_sum[idx], 'env_reward25': env_sum[idx], 'dynamics_term': dynamic_sum[idx], 'head_term': head_error_sum[idx], 'total_reward_bias': total_bias[idx], 'pred_value': value_pred[idx], 'observed_value': value_actual[idx], 'expert_critic': critic[idx], 'value_dynamics_term': value_pred[idx] - value_actual[idx], 'value_head_term': value_actual[idx] - critic[idx], 'action_rms_vs_mean_plan': action_rms[idx], 'command_mae': command_mae[idx], 'raw_csv': str((snapshot_dir / 'candidates.csv').resolve()), 'raw_arrays': str(sequence_file.resolve())}
                branches.append(branch)
                for k in range(25):
                    step_row = {'condition': condition, 'seed': seed, 'snapshot_step': step, 'near_fall_100': int(near), 'candidate_index': idx, 'kind': row['candidate_kind'], 'horizon_step': k + 1, 'pred_reward': predicted[idx, k], 'observed_head_reward': observed_head[idx, k], 'env_reward': environment[idx, k], 'dynamics_term': predicted[idx, k] - observed_head[idx, k], 'head_term': observed_head[idx, k] - environment[idx, k]}
                    steps.append(step_row)
                    for group in ('all', 'near_fall_100' if near else 'other'):
                        horizon_values[condition, group, row['candidate_kind'], k + 1].append((step_row['dynamics_term'], step_row['head_term']))
                for k in range(25):
                    for joint in range(12):
                        action_cells[condition, row['candidate_kind'], k + 1, joint + 1].append((executed[idx, k, joint] - planned[0, k, joint]) ** 2)
            for universe, indices in (('all66', np.arange(66)), ('narrow34', NARROW_INDICES)):
                if not complete[indices].all():
                    continue
                dynamics_rho, dynamics_top8 = rank_pair(pred_sum[indices], head_sum[indices])
                head_rho, head_top8 = rank_pair(head_sum[indices], env_sum[indices])
                ranks.append({'condition': condition, 'seed': seed, 'snapshot_step': step, 'near_fall_100': int(near), 'universe': universe, 'candidate_count': len(indices), 'dynamics_spearman': dynamics_rho, 'dynamics_top8_overlap': dynamics_top8, 'head_spearman': head_rho, 'head_top8_overlap': head_top8})
            if complete[:67].all():
                predicted_total = np.asarray([float(r['pred_total']) for r in candidates[:67]])
                actual_head_total = head_sum[:67] + FINAL_WEIGHT * value_actual[:67]
                rank_pred = score_rank(predicted_total, predicted_total[66])
                rank_head = score_rank(actual_head_total, actual_head_total[66])
                expert_ranks.append({'condition': condition, 'seed': seed, 'snapshot_step': step, 'near_fall_100': int(near), 'predicted_rank_among_67': rank_pred, 'observed_head_rank_among_67': rank_head, 'observed_minus_predicted_rank': rank_head - rank_pred, 'expert_predicted_total': predicted_total[66], 'expert_observed_head_total': actual_head_total[66]})
    assert audit['snapshots'] == 371 and audit['complete_branches'] == 24424, audit
    assert audit['max_pred_reward_vs_csv'] < 1e-06, audit
    assert audit['max_head_reward_vs_csv'] < 1e-06, audit
    assert audit['max_env_reward_vs_csv'] < 1e-06, audit
    assert audit['max_decomposition_identity'] < 1e-12, audit
    assert audit['max_command_mae_vs_csv'] < 1e-06, audit
    assert audit['max_observed_value_vs_csv'] < 1e-06, audit
    assert audit['max_predicted_total_vs_csv'] < 1e-06, audit
    decomposition_summary, horizon_summary, value_summary = ([], [], [])
    command_summary, action_summary, rank_summary, expert_rank_summary = ([], [], [], [])
    for condition in CONDITIONS:
        for group in GROUPS:
            for kind in KINDS:
                subset = group_rows([r for r in branches if r['condition'] == condition and r['kind'] == kind], group)
                d = {'condition': condition, 'group': group, 'kind': kind, 'n': len(subset)}
                for field in ('dynamics_term', 'head_term', 'total_reward_bias'):
                    add_stats(d, field, (r[field] for r in subset))
                decomposition_summary.append(d)
                v = {'condition': condition, 'group': group, 'kind': kind, 'n': len(subset)}
                for field in ('value_dynamics_term', 'value_head_term'):
                    add_stats(v, field, (r[field] for r in subset))
                value_summary.append(v)
                a = {'condition': condition, 'group': group, 'kind': kind, 'n': len(subset)}
                add_stats(a, 'action_rms_vs_mean_plan', (r['action_rms_vs_mean_plan'] for r in subset))
                action_summary.append(a)
                cmd = {'condition': condition, 'group': group, 'kind': kind, 'n': len(subset)}
                add_stats(cmd, 'command_mae', (r['command_mae'] for r in subset))
                x = [r['command_mae'] for r in subset]
                y = [r['total_reward_bias'] for r in subset]
                cmd['bias_pearson'] = pearson(x, y)
                cmd['bias_spearman'] = rho(x, y)
                snap = defaultdict(list)
                for r in subset:
                    snap[r['seed'], r['snapshot_step']].append(r)
                xd, yd = ([], [])
                for rows in snap.values():
                    if len(rows) < 2:
                        continue
                    mx = np.mean([r['command_mae'] for r in rows])
                    my = np.mean([r['total_reward_bias'] for r in rows])
                    xd.extend((r['command_mae'] - mx for r in rows))
                    yd.extend((r['total_reward_bias'] - my for r in rows))
                cmd['within_snapshot_bias_pearson'] = pearson(xd, yd)
                command_summary.append(cmd)
                for k in range(1, 26):
                    rows = horizon_values[condition, group, kind, k]
                    h = {'condition': condition, 'group': group, 'kind': kind, 'horizon_step': k, 'n': len(rows)}
                    add_stats(h, 'dynamics_term', (r[0] for r in rows))
                    add_stats(h, 'head_term', (r[1] for r in rows))
                    horizon_summary.append(h)
            for universe in ('all66', 'narrow34'):
                subset = group_rows([r for r in ranks if r['condition'] == condition and r['universe'] == universe], group)
                q = {'condition': condition, 'group': group, 'universe': universe, 'snapshots': len(subset)}
                for field in ('dynamics_spearman', 'dynamics_top8_overlap', 'head_spearman', 'head_top8_overlap'):
                    add_stats(q, field, (r[field] for r in subset))
                rank_summary.append(q)
            subset = group_rows([r for r in expert_ranks if r['condition'] == condition], group)
            q = {'condition': condition, 'group': group, 'snapshots': len(subset)}
            for field in ('predicted_rank_among_67', 'observed_head_rank_among_67', 'observed_minus_predicted_rank'):
                add_stats(q, field, (r[field] for r in subset))
            expert_rank_summary.append(q)
    action_cells_rows = []
    for (condition, kind, step, joint), squared in sorted(action_cells.items()):
        action_cells_rows.append({'condition': condition, 'kind': kind, 'horizon_step': step, 'joint_index': joint, 'n': len(squared), 'rms': float(np.sqrt(np.mean(squared)))})
    for name, rows in (('branch_decomposition.csv', branches), ('step_decomposition.csv', steps), ('decomposition_summary.csv', decomposition_summary), ('horizon_summary.csv', horizon_summary), ('rank_snapshot.csv', ranks), ('rank_summary.csv', rank_summary), ('expert_rank_snapshot.csv', expert_ranks), ('expert_rank_summary.csv', expert_rank_summary), ('action_summary.csv', action_summary), ('action_rms_step_joint.csv', action_cells_rows), ('command_summary.csv', command_summary), ('value_summary.csv', value_summary)):
        write_csv(name, rows)
    import json
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))
if __name__ == '__main__':
    main()
