"""Descriptive candidate-ranking and branch outcome analysis."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import csv
import json
import math
import statistics as st
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
ROOT = artifact_dir('candidate_rank_v1')
CONDITIONS = ('cp30000_mass8', 'm0_mass0', 'm0_mass8')
GROUPS = ('near_fall_100', 'other')
parser = argparse.ArgumentParser()
parser.add_argument('--input-root', type=Path, default=ROOT / 'full')
parser.add_argument('--output-root', type=Path, default=ROOT / 'analysis')
parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
parser.add_argument('--seeds', type=int, nargs='+', default=list(range(23601, 23609)))
args = parser.parse_args()
INPUT = args.input_root
OUT = args.output_root

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def f(row, key):
    return float(row[key]) if row[key] not in ('', None) else math.nan

def mean_sd(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return (st.mean(values), st.stdev(values) if len(values) > 1 else math.nan, len(values)) if values else (math.nan, math.nan, 0)

def rho(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan
    return float(spearmanr(x, y).statistic)

def pearson(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])

def auc_risk(labels, score):
    labels = np.asarray(labels, bool)
    score = np.asarray(score, float)
    positive, negative = (score[labels], score[~labels])
    if len(positive) == 0 or len(negative) == 0:
        return math.nan
    ranks = rankdata(-score, method='average')
    return float((np.sum(ranks[labels]) - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))

def summarize(rows, key, names):
    mean, sd, n = mean_sd((f(row, key) for row in rows))
    return {**names, 'metric': key, 'n': n, 'mean': mean, 'sample_sd': sd}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_snapshot = []
    candidate_records = []
    optimism = []
    rank_summary = []
    value_summary = []
    state_summary = []
    survival_summary = []
    audit = []
    for condition in args.conditions:
        all_snapshots = []
        episodes = []
        for seed in args.seeds:
            path = INPUT / condition / f'seed_{seed}'
            summary = json.loads((path / 'summary.json').read_text())
            assert summary['seed'] == seed and summary['candidate_count_per_snapshot'] == 66
            assert summary['snapshots_processed'] == summary['snapshots_total']
            episodes.append(summary)
            for snap_path in sorted(path.glob('snapshot_????')):
                snap = json.loads((snap_path / 'summary.json').read_text())
                rows = list(csv.DictReader((snap_path / 'candidates.csv').open()))
                assert len(rows) == 68 and [int(x['candidate_index']) for x in rows] == list(range(68))
                assert snap['branches'] == 68
                candidate_records.extend(rows)
                all_snapshots.append((snap, rows))
                group = 'near_fall_100' if int(rows[0]['near_main_fall_100']) else 'other'
                open_rows = rows[:66]
                pred_r = np.asarray([f(row, 'pred_reward_discounted') for row in open_rows])
                actual_r = np.asarray([f(row, 'actual_reward25_discounted') for row in open_rows])
                pred_t = np.asarray([f(row, 'pred_total') for row in open_rows])
                actual_t = np.asarray([f(row, 'actual_total_with_critic') for row in open_rows])
                pred_top8_r = set(np.argsort(-pred_r)[:8].tolist())
                actual_top8_r = set(np.argsort(-actual_r)[:8].tolist())
                pred_top8_t = set(np.argsort(-pred_t)[:8].tolist())
                actual_top8_t = set(np.argsort(-actual_t)[:8].tolist())
                local_idx = np.asarray(list(range(33)) + [65])
                local_pr, local_ar = (pred_r[local_idx], actual_r[local_idx])
                local_pt, local_at = (pred_t[local_idx], actual_t[local_idx])
                expert, base = (rows[66], rows[67])
                expert_rank = int(1 + sum(pred_t > f(expert, 'pred_total'))) if int(expert['expert_action_sequence_complete']) else math.nan
                record = {'condition': condition, 'seed': seed, 'snapshot_step': snap['step'], 'group': group, 'steps_to_main_fall': snap['steps_to_main_fall'] or '', 'reward_spearman': rho(pred_r.tolist(), actual_r.tolist()), 'reward_top8_overlap': len(pred_top8_r & actual_top8_r), 'reward_regret': float(np.max(actual_r) - actual_r[np.argmax(pred_r)]), 'total_spearman': rho(pred_t.tolist(), actual_t.tolist()), 'total_top8_overlap': len(pred_top8_t & actual_top8_t), 'total_regret': float(np.max(actual_t) - actual_t[np.argmax(pred_t)]), 'local_reward_spearman': rho(local_pr.tolist(), local_ar.tolist()), 'local_reward_regret': float(np.max(local_ar) - local_ar[np.argmax(local_pr)]), 'local_total_spearman': rho(local_pt.tolist(), local_at.tolist()), 'local_total_regret': float(np.max(local_at) - local_at[np.argmax(local_pt)]), 'expert_margin25': f(expert, 'actual_reward25_discounted') - f(rows[0], 'actual_reward25_discounted'), 'expert_pred_total_rank_among_67': expert_rank, 'expert_survived200': int(expert['actual_fell'] == '0' and int(expert['actual_steps']) == 200), 'base_survived200': int(base['actual_fell'] == '0' and int(base['actual_steps']) == 200), 'expert_return200': f(expert, 'actual_reward200_discounted'), 'base_return200': f(base, 'actual_reward200_discounted'), 'snapshot_head_value_raw': f(rows[0], 'snapshot_head_value_raw'), 'snapshot_expert_critic': f(rows[0], 'snapshot_expert_critic'), 'max_restore_root_error': snap['max_restore_root_error'], 'max_restore_joint_error': snap['max_restore_joint_error'], 'mean_first_reward_error': snap['mean_first_reward_error']}
                per_snapshot.append(record)
        audit.append({'condition': condition, 'episodes': len(episodes), 'main_falls': sum((int(ep['main_fell']) for ep in episodes)), 'snapshots': len(all_snapshots), 'candidate_branches': len(all_snapshots) * 66, 'closed_branches': len(all_snapshots) * 2, 'main_steps': sum((ep['main_episode_length'] for ep in episodes)), 'runtime_seconds': sum((ep['duration_seconds'] for ep in episodes)), 'max_first_reward_error': max((snap['mean_first_reward_error'] for snap, _ in all_snapshots)), 'max_restore_root_error': max((snap['max_restore_root_error'] for snap, _ in all_snapshots)), 'max_restore_joint_error': max((snap['max_restore_joint_error'] for snap, _ in all_snapshots))})
        for group in GROUPS:
            selected_snapshots = [(snap, rows) for snap, rows in all_snapshots if ('near_fall_100' if int(rows[0]['near_main_fall_100']) else 'other') == group]
            selected = [row for row in per_snapshot if row['condition'] == condition and row['group'] == group]
            for key in ('reward_spearman', 'reward_top8_overlap', 'reward_regret', 'total_spearman', 'total_top8_overlap', 'total_regret', 'local_reward_spearman', 'local_reward_regret', 'local_total_spearman', 'local_total_regret', 'expert_margin25', 'expert_pred_total_rank_among_67'):
                rank_summary.append(summarize(selected, key, {'condition': condition, 'group': group}))
            survival_summary.append({'condition': condition, 'group': group, 'snapshots': len(selected), 'expert_survived200': sum((r['expert_survived200'] for r in selected)), 'expert_survival_rate': st.mean((r['expert_survived200'] for r in selected)) if selected else math.nan, 'base_survived200': sum((r['base_survived200'] for r in selected)), 'base_survival_rate': st.mean((r['base_survived200'] for r in selected)) if selected else math.nan})
            flat = [row for _, rows in selected_snapshots for row in rows]
            for kind in ('mean_plan', 'last_iteration', 'noise_0.1', 'noise_0.3', 'noise_1.0', 'base_proposal', 'expert_closed', 'base_closed'):
                subset = [row for row in flat if row['candidate_kind'] == kind]
                errors = [f(row, 'pred_reward_discounted') - f(row, 'actual_reward25_discounted') for row in subset]
                mean, sd, n = mean_sd(errors)
                optimism.append({'condition': condition, 'group': group, 'kind': kind, 'n': n, 'mean_pred_minus_actual_reward25': mean, 'sample_sd': sd, 'early_falls_under25': sum((int(r['actual_fell']) and int(r['actual_steps']) < 25 for r in subset)), 'mean_command_mae': st.mean((f(row, 'command_mae') for row in subset)) if subset else math.nan})
            full = [row for row in flat if int(row['actual_25_complete'])]
            predicted = [f(row, 'pred_terminal_value_raw') for row in full]
            head = [f(row, 'actual_head_terminal_value_raw') for row in full]
            critic = [f(row, 'expert_critic_step25') for row in full]
            diff = [p - h for p, h in zip(predicted, head)]
            value_summary.append({'condition': condition, 'group': group, 'n_full25': len(full), 'pred_minus_actual_head_value_mean': mean_sd(diff)[0], 'pred_minus_actual_head_value_sample_sd': mean_sd(diff)[1], 'pred_vs_actual_head_value_pearson': pearson(predicted, head), 'pred_vs_actual_head_value_spearman': rho(predicted, head), 'actual_head_vs_expert_critic_spearman': rho(head, critic)})
            for predictor in ('snapshot_head_value_raw', 'snapshot_expert_critic'):
                x = [row[predictor] for row in selected]
                state_summary.append({'condition': condition, 'group': group, 'predictor': predictor, 'snapshots': len(selected), 'spearman_expert_return200': rho(x, [row['expert_return200'] for row in selected]), 'spearman_base_return200': rho(x, [row['base_return200'] for row in selected]), 'base_fall_count': sum((1 - row['base_survived200'] for row in selected)), 'base_fall_auc': auc_risk([1 - row['base_survived200'] for row in selected], x)})
    write_csv(OUT / 'audit.csv', audit)
    write_csv(OUT / 'candidate_records.csv', candidate_records)
    write_csv(OUT / 'per_snapshot_metrics.csv', per_snapshot)
    write_csv(OUT / 'ranking_summary.csv', rank_summary)
    write_csv(OUT / 'optimism_summary.csv', optimism)
    write_csv(OUT / 'value_summary.csv', value_summary)
    write_csv(OUT / 'state_value_summary.csv', state_summary)
    write_csv(OUT / 'survival_summary.csv', survival_summary)
    for row in audit:
        print('AUDIT', json.dumps(row), flush=True)
if __name__ == '__main__':
    main()
