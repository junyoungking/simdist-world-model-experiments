"""Recalculate Return/ET/paired evidence and critic support without simulator."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = artifact_dir('privileged_ppo')
REPO = PORTFOLIO_ROOT

def csvread(path):
    return list(csv.DictReader(resolve_artifact_path(path).open()))

def csvwrite(path, rows):
    with resolve_artifact_path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def read(path):
    return json.loads(resolve_artifact_path(path).read_text())

def mppi_rows(category, models, seeds):
    rows = []
    for model in models:
        for seed in seeds:
            name = f'{model}_{seed}'
            data = ROOT / category / 'mppi/datasets' / name
            roll = read(data / 'rollout.json')
            full = read(data / 'full_runtime_readback.json')[0]
            raw_paths = sorted((ROOT / category / f'raw_{name}' / 'raw').glob('rollout_episode_*.npz'))
            rewards = []
            vx = []
            actions = []
            for p in raw_paths:
                with np.load(p) as raw:
                    rewards.append(raw['reward_t'])
                    vx.append(raw['proprio_obs'][:, 0])
                    actions.append(raw['action'])
            rewards = np.concatenate(rewards)
            vx = np.concatenate(vx)
            actions = np.concatenate(actions)
            assert len(rewards) == 1000 and np.isfinite(rewards).all()
            et = sum((x['end_reason'] == 'terminated' for x in roll['episodes']))
            timeout = sum((x['end_reason'] == 'timeout' for x in roll['episodes']))
            with np.load(data / 'actuator_trace.npz') as trace:
                saturation = float(np.mean(np.abs(trace['computed_torque'] - trace['applied_torque']) > 1e-06))
            rows.append({'model': model, 'train_seed': '', 'eval_seed': seed, 'Return': float(rewards.sum(dtype=np.float64)), 'ET_seed': int(et > 0), 'ET_events': et, 'executed_steps': len(rewards), 'segments': len(roll['episodes']), 'timeout_count': timeout, 'automatic_reset_count': et + timeout, 'budget_censored_segments': sum((x['end_reason'] == 'budget_censored' for x in roll['episodes'])), 'zero_action_warmup_steps': sum((x['stabilization_rows'] for x in roll['episodes'])), 'mean_action_norm': float(np.linalg.norm(actions, axis=1).mean()), 'torque_saturation_fraction': saturation, 'vx_RMSE': float(np.sqrt(np.mean((vx.astype(np.float64) - 0.5) ** 2))), 'checkpoint': roll['source_checkpoint'], 'checkpoint_sha256': read(document_path('privileged_ppo', 'CHECKPOINT_HASHES.json'))[model]['bundle_sha256'], 'terrain_sha256': full['terrain_mesh_sha256'], 'initial_state_sha256': full['initial_root_joint_state_sha256'], 'reward_signature': full['reward_signature'], 'termination_signature': full['termination_signature'], 'action_signature': full['action_signature'], 'startup_zero_action_steps': 25, 'raw_directory': str(ROOT / category / f'raw_{name}' / 'raw')})
    return rows

def ppo_rows(category, model):
    rows = []
    for row in csvread(ROOT / category / model / 'EVALUATION_RAW.csv'):
        with np.load(row['raw_path']) as raw:
            ret = float(raw['reward'].sum(dtype=np.float64))
            et = int(raw['terminated'].sum())
            assert abs(ret - float(row['Return'])) < 1e-10 and et == int(row['ET_events'])
            saturation = float(np.mean(np.abs(raw['computed_torque'] - raw['applied_torque']) > 1e-06))
        rows.append({'model': model, 'train_seed': model.split('_')[-1] if model.startswith('PPO_') else '', 'eval_seed': int(row['eval_seed']), 'Return': ret, 'ET_seed': int(et > 0), 'ET_events': et, 'executed_steps': int(row['executed_steps']), 'segments': int(row['segments']), 'timeout_count': int(row['timeout_count']), 'automatic_reset_count': et + int(row['timeout_count']), 'budget_censored_segments': int(row['segments']) - et - int(row['timeout_count']), 'zero_action_warmup_steps': 0, 'mean_action_norm': float(row['mean_action_norm']), 'torque_saturation_fraction': saturation, 'vx_RMSE': float(row['vx_RMSE']), 'checkpoint': row['checkpoint'], 'checkpoint_sha256': row['checkpoint_sha256'], 'terrain_sha256': row['terrain_sha256'], 'initial_state_sha256': row['initial_state_sha256'], 'reward_signature': row['reward_signature'], 'termination_signature': row['termination_signature'], 'action_signature': row['action_signature'], 'startup_zero_action_steps': 0, 'raw_directory': str(ROOT / category / model)})
    return rows
primary = mppi_rows('05_mass8_primary', ['M0', 'MB1', 'MB2', 'MB3'], range(61001, 61011))
for seed in [62001, 62002, 62003]:
    primary.extend(ppo_rows('05_mass8_primary', f'PPO_{seed}'))
zero = mppi_rows('04_0kg_anchor', ['M0'], range(66001, 66011)) + ppo_rows('04_0kg_anchor', 'source_PPO')
assert len(primary) == 70 and len({(x['model'], x['eval_seed']) for x in primary}) == 70
assert len(zero) == 20
csvwrite(ROOT / '05_mass8_primary/MASS8_PRIMARY_EVAL_RAW.csv', primary)
csvwrite(ROOT / '04_0kg_anchor/0KG_SOURCE_ANCHOR.csv', zero)
summaries = []
for condition, rows in [('mass8', primary), ('mass0', zero)]:
    for model in dict.fromkeys((x['model'] for x in rows)):
        group = [x for x in rows if x['model'] == model]
        vals = np.array([x['Return'] for x in group])
        mean = vals.mean()
        sd = vals.std(ddof=1)
        half = stats.t.ppf(0.975, 9) * sd / np.sqrt(10)
        successes = sum((x['ET_seed'] for x in group))
        p = successes / 10.0
        z = stats.norm.ppf(0.975)
        mid = (p + z * z / 20) / (1 + z * z / 10)
        err = z * np.sqrt(p * (1 - p) / 10 + z * z / 400) / (1 + z * z / 10)
        summaries.append({'condition': condition, 'model': model, 'n_eval_seeds': 10, 'mean_Return': mean, 'SD_Return': sd, 'descriptive_t95_low': mean - half, 'descriptive_t95_high': mean + half, 'ET_seeds': successes, 'ET_events': sum((x['ET_events'] for x in group)), 'ET_Wilson_low': mid - err, 'ET_Wilson_high': mid + err, 'mean_vx_RMSE': np.mean([x['vx_RMSE'] for x in group]), 'executed_steps': sum((x['executed_steps'] for x in group))})
csvwrite(ROOT / '07_summary/MODEL_SEED_SUMMARY.csv', summaries)
repeat_summaries = []
for label, models in [('PPO_training_seeds', ['PPO_62001', 'PPO_62002', 'PPO_62003']), ('MB_adaptation_checkpoints', ['MB1', 'MB2', 'MB3'])]:
    group = [s for s in summaries if s['condition'] == 'mass8' and s['model'] in models]
    values = np.array([s['mean_Return'] for s in group])
    repeat_summaries.append({'repeat_type': label, 'n_model_repeats': 3, 'mean_of_eval_means': values.mean(), 'SD_of_eval_means': values.std(ddof=1), 'min_eval_mean': values.min(), 'max_eval_mean': values.max(), 'pooled_ET_cells_descriptive': sum((s['ET_seeds'] for s in group)), 'eval_cells': 30, 'caveat': 'Distinct repeat types; evaluation cells are not independent trained policies'})
csvwrite(ROOT / '07_summary/REPEAT_VARIANCE_SUMMARY.csv', repeat_summaries)
contrasts = []
for condition, rows in [('mass8', primary), ('mass0', zero)]:
    models = list(dict.fromkeys((x['model'] for x in rows)))
    pairs = [('M0', m) for m in models if m != 'M0']
    if condition == 'mass8':
        pairs += [(m, p) for m in ['MB1', 'MB2', 'MB3'] for p in models if p.startswith('PPO_')]
    for a, b in pairs:
        for seed in sorted({r['eval_seed'] for r in rows}):
            ra = next((r for r in rows if r['model'] == a and r['eval_seed'] == seed))
            rb = next((r for r in rows if r['model'] == b and r['eval_seed'] == seed))
            contrasts.append({'condition': condition, 'reference': a, 'comparison': b, 'eval_seed': seed, 'reference_Return': ra['Return'], 'comparison_Return': rb['Return'], 'comparison_minus_reference': rb['Return'] - ra['Return'], 'reference_ET': ra['ET_seed'], 'comparison_ET': rb['ET_seed']})
csvwrite(ROOT / '07_summary/PAIRED_RETURN_DELTAS.csv', contrasts)
paired_summary = []
for condition, a, b in dict.fromkeys(((r['condition'], r['reference'], r['comparison']) for r in contrasts)):
    delta = np.array([x['comparison_minus_reference'] for x in contrasts if (x['condition'], x['reference'], x['comparison']) == (condition, a, b)])
    mean = delta.mean()
    sd = delta.std(ddof=1)
    half = stats.t.ppf(0.975, 9) * sd / np.sqrt(10)
    paired_summary.append({'condition': condition, 'reference': a, 'comparison': b, 'mean_delta': mean, 'SD_delta': sd, 'descriptive_t95_low': mean - half, 'descriptive_t95_high': mean + half})
csvwrite(ROOT / '07_summary/PAIRED_DELTA_SUMMARY.csv', paired_summary)
convergence = []
curves = []
train_diagnostics = []
for seed in [62001, 62002, 62003]:
    result = read(ROOT / '02_training' / f'seed_{seed}' / 'result.json')
    convergence.append(result)
    curves.extend(csvread(ROOT / '02_training' / f'seed_{seed}' / 'PPO_TRAINING_CURVES.csv'))
    iterations = csvread(ROOT / '02_training' / f'seed_{seed}' / 'TRAIN_ITERATIONS.csv')
    assert len(iterations) == result['target_iterations']
    assert [int(r['target_iteration']) for r in iterations] == list(range(1, len(iterations) + 1))
    final_window = iterations[-250:]
    for field in ['value_loss', 'KL_full_update', 'mean_action_std', 'min_action_std', 'entropy']:
        values = np.array([float(r[field]) for r in final_window])
        if field == 'entropy':
            values = values / 20
        train_diagnostics.append({'train_seed': seed, 'metric': field if field != 'entropy' else 'mean_minibatch_differential_entropy', 'final_window_iterations': 250, 'window_mean': values.mean(), 'window_min': values.min(), 'window_max': values.max(), 'last_value': values[-1], 'all_training_values_finite': all((np.isfinite(float(r[field])) for r in iterations))})
csvwrite(ROOT / '07_summary/PPO_TRAINING_CURVES.csv', curves)
csvwrite(ROOT / '07_summary/PPO_TRAIN_DIAGNOSTICS.csv', train_diagnostics)
training_manifest = []
for r in convergence:
    training_manifest.append({'train_seed': r['train_seed'], 'initialization': 'A_actor_critic_optimizer_warm_start', 'actor_dim': 433, 'critic_dim': 433, 'normalizer': 'Identity', 'num_envs': 4096, 'rollout_steps': 24, 'target_iterations': r['target_iterations'], 'target_env_steps': r['target_env_steps'], 'target_optimizer_updates': r['target_optimizer_updates'], 'selection': 'operational_plateau_final', 'final_checkpoint': r['final']['checkpoint'], 'final_sha256': r['final']['checkpoint_sha256'], 'status': r['status'], 'source_checkpoint_sha256': '48cf4e95ef853d2addc5c3a5b2e2eab675dfc5d0b89d5ca33efc6aa102857d7a'})
csvwrite(ROOT / '02_training/PPO_TRAIN_MANIFEST.csv', training_manifest)
critic_rows = []
for seed in [62001, 62002, 62003]:
    for row in csvread(ROOT / '06_critic' / f'PPO_{seed}' / 'EVALUATION_RAW.csv'):
        eval_seed = int(row['eval_seed'])
        with np.load(row['raw_path']) as raw:
            reward = raw['reward'].astype(np.float64)
            value = raw['critic_value']
            episodes = raw['episode']
            terminal = raw['terminated']
            for episode in np.unique(episodes):
                ids = np.flatnonzero(episodes == episode)
                complete = bool(terminal[ids[-1]])
                if complete:
                    ret = 0.0
                    for idx in ids[::-1]:
                        ret = reward[idx] + 0.99 * ret
                        critic_rows.append({'train_seed': seed, 'eval_seed': eval_seed, 'episode': int(episode), 'step': int(idx), 'target_type': 'complete_non_timeout_MC', 'V': float(value[idx]), 'discounted_RTG': ret, 'remaining_steps': int(ids[-1] - idx + 1)})
                else:
                    for offset in range(max(0, len(ids) - 500 + 1)):
                        window = ids[offset:offset + 500]
                        ret = float((reward[window] * 0.99 ** np.arange(500)).sum())
                        idx = window[0]
                        critic_rows.append({'train_seed': seed, 'eval_seed': eval_seed, 'episode': int(episode), 'step': int(idx), 'target_type': 'truncated_500_no_bootstrap_secondary', 'V': float(value[idx]), 'discounted_RTG': ret, 'remaining_steps': 500})
csvwrite(ROOT / '06_critic/CRITIC_VALUE_RTG_RAW.csv', critic_rows)
critic_summary = []
for seed in [62001, 62002, 62003]:
    for target in ['complete_non_timeout_MC', 'truncated_500_no_bootstrap_secondary']:
        rows = [r for r in critic_rows if r['train_seed'] == seed and r['target_type'] == target]
        if rows:
            v = np.array([r['V'] for r in rows])
            mc = np.array([r['discounted_RTG'] for r in rows])
            err = v - mc
            rho = stats.spearmanr(v, mc).statistic if np.ptp(v) > 0 and np.ptp(mc) > 0 else None
            critic_summary.append({'train_seed': seed, 'target_type': target, 'n_supported_states': len(rows), 'bias': float(err.mean()), 'RMSE': float(np.sqrt(np.mean(err * err))), 'Spearman_rho': rho})
        else:
            critic_summary.append({'train_seed': seed, 'target_type': target, 'n_supported_states': 0, 'bias': 'UNAVAILABLE', 'RMSE': 'UNAVAILABLE', 'Spearman_rho': 'UNAVAILABLE'})
csvwrite(ROOT / '06_critic/CRITIC_QUALITY_SUMMARY.csv', critic_summary)
pairing = []
for condition, rows in [('mass8', primary), ('mass0', zero)]:
    for seed in sorted({r['eval_seed'] for r in rows}):
        group = [r for r in rows if r['eval_seed'] == seed]
        checks = {k: len({r[k] for r in group}) == 1 for k in ['terrain_sha256', 'initial_state_sha256', 'reward_signature', 'termination_signature', 'action_signature']}
        pairing.append({'condition': condition, 'eval_seed': seed, **checks})
csvwrite(ROOT / '07_summary/PAIRED_RUNTIME_CONTRACT_AUDIT.csv', pairing)
assert all((all((r[k] for k in ['terrain_sha256', 'initial_state_sha256', 'reward_signature', 'termination_signature', 'action_signature'])) for r in pairing)), 'Runtime pairing mismatch must be resolved/reported before completion'
roles = read(document_path('privileged_ppo', 'SEEDS.json'))['seed_roles']
leakage = []
for role in ['train', 'validation', 'zero_kg', 'critic', 'smoke']:
    overlap = sorted(set(roles['primary']) & set(roles[role]))
    leakage.append({'comparison': 'primary_vs_' + role, 'overlap_count': len(overlap), 'status': 'PASS' if not overlap else 'FAIL', 'scope': 'Frozen new execution roles; historical source PPO/WM seed logs unavailable'})
csvwrite(ROOT / '07_summary/HELDOUT_LEAKAGE_AUDIT.csv', leakage)
assert all((r['status'] == 'PASS' for r in leakage))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, condition, rows in zip(axes, ['mass0', 'mass8'], [zero, primary]):
    models = list(dict.fromkeys((x['model'] for x in rows)))
    for i, model in enumerate(models):
        group = [x for x in rows if x['model'] == model]
        for j, r in enumerate(group):
            ax.scatter(i + (j - 4.5) * 0.035, r['Return'], marker='X' if r['ET_seed'] else 'o', s=35)
        ax.hlines(np.mean([x['Return'] for x in group]), i - 0.25, i + 0.25, color='black')
        s = next((s for s in summaries if s['condition'] == condition and s['model'] == model))
        ax.errorbar(i, s['mean_Return'], yerr=[[s['mean_Return'] - s['descriptive_t95_low']], [s['descriptive_t95_high'] - s['mean_Return']]], color='black', capsize=4, fmt='none', linewidth=1.3)
    ax.set_xticks(range(len(models)), [m + '\nET ' + str(sum((x['ET_seed'] for x in rows if x['model'] == m))) + '/10' for m in models], rotation=25)
    ax.set_title('0 kg source anchor' if condition == 'mass0' else '+8 kg primary: privileged PPO reference')
    ax.set_ylabel('Return / 1000 executed steps')
    ax.grid(axis='y', alpha=0.2)
axes[1].legend(handles=[Line2D([], [], marker='o', color='tab:blue', linestyle='none', label='No non-timeout ET'), Line2D([], [], marker='X', color='tab:blue', linestyle='none', label='At least one non-timeout ET')], fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / '07_summary/PERFORMANCE_GAP_PLOT.png', dpi=180)
plt.close(fig)
fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
for seed in [62001, 62002, 62003]:
    history = read(ROOT / '02_training' / f'seed_{seed}' / 'VALIDATION_SUMMARY.json')
    steps = np.array([x['target_env_steps'] for x in history])
    means = np.array([x['mean_Return'] for x in history])
    sd = np.array([x['SD_Return'] for x in history])
    axes[0].plot(steps, means, label=str(seed))
    axes[0].fill_between(steps, means - sd, means + sd, alpha=0.12)
    axes[1].plot(steps, [x['ET_seed_rate'] for x in history])
    axes[2].plot(steps, [x['mean_vx_RMSE'] for x in history])
axes[0].set_ylabel('Validation Return ± eval-seed SD')
axes[0].legend(title='Training seed')
axes[1].set_ylabel('ET seed rate')
axes[2].set_ylabel('vx RMSE (m/s)')
axes[2].set_xlabel('Target environment transitions')
fig.tight_layout()
fig.savefig(ROOT / '07_summary/PPO_VALIDATION_LEARNING_CURVES.png', dpi=180)
plt.close(fig)
ppo_repeat = repeat_summaries[0]
mb_repeat = repeat_summaries[1]
lines = ['# +8kg privileged PPO target capability comparison', '', f"Fresh +8 kg tests give the three privileged target PPOs a mean-of-model-means Return of {ppo_repeat['mean_of_eval_means']:.6f} (between-training-seed SD {ppo_repeat['SD_of_eval_means']:.6f}); MB1–3 have {mb_repeat['mean_of_eval_means']:.6f} (between-adaptation-checkpoint SD {mb_repeat['SD_of_eval_means']:.6f}). The descriptive PPO-minus-MB average difference is {ppo_repeat['mean_of_eval_means'] - mb_repeat['mean_of_eval_means']:+.6f} per 1000 executed steps. All three policies satisfied the predeclared operational validation plateau and additional-training rule. This supports an observed target-domain performance comparison within the measured support, not optimality or a frozen-R/V causal claim.", '', 'PPO A: source actor+critic+optimizer warm-start,433 privileged dimensions. This is an attainable reference, not an optimum or frozen-R/V attribution. PPO begins control immediately; MPPI retains25 zero-action steps. Historical23501–23505 are separate from fresh61001–61010.', '', '| condition | controller/model | mean Return | SD | descriptive t95 interval | ET seeds/10 | ET events | vx RMSE |', '|---|---|---:|---:|---|---:|---:|---:|']
for s in summaries:
    lines.append(f"| {s['condition']} | {s['model']} | {s['mean_Return']:.6f} | {s['SD_Return']:.6f} | [{s['descriptive_t95_low']:.6f},{s['descriptive_t95_high']:.6f}] | {s['ET_seeds']}/10 | {s['ET_events']} | {s['mean_vx_RMSE']:.6f} |")
lines += ['', '## Source, physics and interface contracts', '', f"Scientific protocol commit `{read(document_path('privileged_ppo', 'PROTOCOL_HASHES.json'))['git_commit']}`, SHA256 `{read(document_path('privileged_ppo', 'PROTOCOL_HASHES.json'))['protocol_sha256']}`. Original source HEAD `{(ROOT / '09_provenance/GIT_HEAD.txt').read_text().strip()}`; pre-existing MPPI/trainer dirty diff is retained, not silently reset. New-root code is committed separately and hashed in09_provenance. Current versions: IsaacSim4.5.0.0, IsaacLab source VERSION2.0.1 (installed metadata0.33.17), Python3.10.21, Torch2.5.1+cu121/CUDA build12.1, RSL-RL2.2.1, RTX4070SUPER12282MiB, NVIDIA driver580.178.04. Historical version equivalence is not established.", '', 'Runtime target base mass6.921000480651855→14.921000480651855kg; default other-body masses remain unchanged and inertia scales with the additive mass ratio. Every initial/reset readback checks Kp25/Kd0.5, four-foot static/dynamic friction0.8000000119/restitution0.05000000075, effort limit23.5 and velocity limit30. Control dt0.02, physics dt0.005; boxes terrain difficulty0.4. WalkStraightCommand advances the0.5m/s reference trajectory and uses position/heading feedback, so its instantaneous command can differ from(0.5,0,0). Initial/reset command fields and terrain/full root+joint-state hashes are retained. Background joint-friction Uniform(0,0.05) additive reset remains canonical; source gain/material/base-pose/push/curriculum randomization is fixed or removed as specified in effective configs.', '', 'Both controllers emit12 raw joint-position-action coordinates in source joint order FR/FL/RR/RL hip/thigh/calf; physical target=default joint position+0.25×raw action, configured clip None. PPO actor/critic are433→512ELU→512ELU→512ELU→12/1 with Identity normalizers. Privileged fields add24 foot-wrench,4 foot-height,1 mass,3 material,12 stiffness,12 damping,12 joint-friction dimensions to365 other dimensions. SimDist inputs are35 proprioception,315 terrain scan,3 command,12 action coordinates over historyH25; latent64. MPPI uses horizonT25,450 noise+22 mixture candidates,8 iterations,64 elites, discount0.99; its predicted score is distinct from actual accumulated evaluation reward. It executes25 zero actions after every reset while the environment reward still uses its feedback command. PPO acts immediately; this startup/controller/information difference prevents an observation-matched or isolated-value causal interpretation.', '', '| checkpoint | source/model step | SHA256 bundle or file |', '|---|---:|---|']
for model, cp in read(document_path('privileged_ppo', 'CHECKPOINT_HASHES.json')).items():
    if model in ['M0', 'MB1', 'MB2', 'MB3']:
        lines.append(f"| {model} | {(367004 if model == 'M0' else 368004)} | `{cp['bundle_sha256']}` |")
lines += ['| released source PPO | retained iter4999; original source env-step provenance unavailable | `48cf4e95ef853d2addc5c3a5b2e2eab675dfc5d0b89d5ca33efc6aa102857d7a` |', '', 'Per-file hash inventory and bundle definition are in CHECKPOINT_HASHES.json. CPU checkpoint-state comparison confirms each MB changes52 learned dynamics leaves plus1 RNG counter and0 leaves outside dynamics; MB f-onlyK1000/batch32 provenance is reused unchanged. Source checkpoints and402 audited code/config/raw files are rehashed against the prior audit. Target PPO counters count only added target training, not source iter4999 or excluded smoke transitions. Training4096×24×5000=491520000 is the source recipe reference milestone from [paper Appendix C](https://arxiv.org/html/2603.15759v1), not a target convergence proof or compulsory stopping iteration.', '', 'Reward definitions are exactly the retained canonical dictionaries and applied on the same control dt. PPO step term contributions are raw manager reward densities×0.02; their sum is checked against actual recorded reward. Evaluation Return uniformly sums float32 recorded samples in float64. Non-timeout terminated is ET; timeout and budget censoring are separate, all runs fill1000 executed steps. Training RSL timeout bootstrap is disclosed separately from raw evaluation reward. Tracking RMSE uses actual pre-action body forward vx against reference0.5 across all1000 steps, including MPPI warmup. The t95 intervals condition on a fixed model and ten evaluation seeds. Torque clipping fraction uses post-step computed-versus-applied torque differences (>1e-6) on joint/control-step cells; reset-step trace semantics are shared.']
lines += ['', '## All raw per-evaluation-seed rows', '', '| condition | model | eval seed | Return | ET0/1 | ET events | vx RMSE | steps |', '|---|---|---:|---:|---:|---:|---:|---:|']
for condition, rows in [('mass8', primary), ('mass0', zero)]:
    for r in rows:
        lines.append(f"| {condition} | {r['model']} | {r['eval_seed']} | {r['Return']:.9f} | {r['ET_seed']} | {r['ET_events']} | {r['vx_RMSE']:.6f} | {r['executed_steps']} |")
lines += ['', '## Paired differences', '', '| condition | comparison minus reference | mean delta | SD | conditional descriptive t95 |', '|---|---|---:|---:|---|']
for r in paired_summary:
    lines.append(f"| {r['condition']} | {r['comparison']}−{r['reference']} | {r['mean_delta']:+.6f} | {r['SD_delta']:.6f} | [{r['descriptive_t95_low']:.6f},{r['descriptive_t95_high']:.6f}] |")
lines += ['', '## Convergence: each independent training seed', '']
for r in convergence:
    c = r['confirmation']
    lines += [f"Seed{r['train_seed']}: {r['target_iterations']} target iterations, {r['target_env_steps']} target transitions, {r['target_optimizer_updates']} target Adam updates. Status {r['status']}. Candidate iteration{c['candidate_iteration']} confirmed after{c['confirmation_iteration'] - c['candidate_iteration']} additional iterations. Paired validation mean change{c['paired_change_mean']:+.6f}, SD{c['paired_change_SD']:.6f}, t95{c['paired_change_t95']}; ET rate change{c['candidate_to_current_ET_rate_change']:+.3f}, vx RMSE change{c['candidate_to_current_vx_RMSE_change']:+.6f}. Final validation Return{r['final']['mean_Return']:.6f}, SD{r['final']['SD_Return']:.6f}, ET rate{r['final']['ET_seed_rate']:.3f}. Final checkpoint `{r['final']['checkpoint']}`, SHA256 `{r['final']['checkpoint_sha256']}`.", '']
lines += ['### Every validation checkpoint', '', '| training seed | target iterations | target transitions | validation mean Return | eval SD | ET events | ET seed rate | vx RMSE |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
for seed in [62001, 62002, 62003]:
    for h in read(ROOT / '02_training' / f'seed_{seed}' / 'VALIDATION_SUMMARY.json'):
        lines.append(f"| {seed} | {h['target_iteration']} | {h['target_env_steps']} | {h['mean_Return']:.6f} | {h['SD_Return']:.6f} | {h['ET_events']} | {h['ET_seed_rate']:.3f} | {h['mean_vx_RMSE']:.6f} |")
lines += ['', '### Training stability, final 250-iteration window', '', '| training seed | metric | mean | min | max | last |', '|---:|---|---:|---:|---:|---:|']
for d in train_diagnostics:
    lines.append(f"| {d['train_seed']} | {d['metric']} | {d['window_mean']:.8g} | {d['window_min']:.8g} | {d['window_max']:.8g} | {d['last_value']:.8g} |")
lines += ['', 'Entropy is corrected for the installed source logging accumulator: the raw source sum across 20 mini-batches is divided by 20 for this table. Negative Gaussian differential entropy alone is not failure; action standard deviations and KL are reported alongside it. All recorded weights/actions/rewards and diagnostic values are checked finite. Value loss is not the policy plateau criterion.', '', '### Repeat variance units', '', '| repeat type | repeats | mean of model means | SD of model means | range | descriptive ET cells |', '|---|---:|---:|---:|---|---:|']
for s in repeat_summaries:
    lines.append(f"| {s['repeat_type']} | 3 | {s['mean_of_eval_means']:.6f} | {s['SD_of_eval_means']:.6f} | [{s['min_eval_mean']:.6f},{s['max_eval_mean']:.6f}] | {s['pooled_ET_cells_descriptive']}/30 |")
lines += ['## Critic support diagnostics', '', '| train seed | target | supported states | bias V−RTG | RMSE | rank correlation |', '|---:|---|---:|---:|---:|---:|']
for r in critic_summary:
    lines.append('| ' + ' | '.join((str(r[k]) for k in ['train_seed', 'target_type', 'n_supported_states', 'bias', 'RMSE', 'Spearman_rho'])) + ' |')
historical = csvread(ROOT / '07_summary/HISTORICAL_S4_ANCHOR.csv')
assert len(historical) == 20
lines += ['', '## Historical corrected S4 consistency, separate provenance', '', '| model | historical eval seed | float32 canonical Return | ET seed | ET events | source config SHA256 |', '|---|---:|---:|---:|---:|---|']
for h in historical:
    lines.append(f"| {h['model']} | {h['eval_seed']} | {float(h['Return_canonical_float32']):.9f} | {h['ET_seed']} | {h['ET_event_count']} | `{h['config_sha256']}` |")
lines += ['', '| model | historical mean Return | paired change vs M0 | ET seeds/5 |', '|---|---:|---:|---:|']
for model, anchor in [('M0', 36.25952), ('MB1', 38.90413), ('MB2', 37.812854), ('MB3', 35.935677)]:
    group = [h for h in historical if h['model'] == model]
    mean = np.mean([float(h['Return_canonical_float32']) for h in group])
    assert abs(mean - anchor) < 1e-06
    m0mean = np.mean([float(h['Return_canonical_float32']) for h in historical if h['model'] == 'M0'])
    lines.append(f"| {model} | {mean:.6f} | {mean - m0mean:+.6f} | {sum((int(h['ET_seed']) for h in group))}/5 |")
lines += ['', 'Critic rollouts sample the final trained Gaussian policy on separate67001–67010; primary/validation use deterministic actor mean. Critic fit therefore targets the trained stochastic policy, not directly the deterministic primary deployment. Complete terminal Monte Carlo fit uses gamma.99 and stops at non-timeout termination. Timeout/censored segments are excluded from this fit. The separate500-step unbootstrapped diagnostic has a truncated target and must not be called exact full-horizon value accuracy. States from one trajectory are correlated; no global oracle claim. Source RSL timeout bootstrap adds gamma times pre-step critic V on truncated transitions; this is recorded separately from raw evaluation reward.', '', '## Units, provenance and limitations', '', 'Within-checkpoint intervals condition on10 evaluation seeds. PPO variation across3 independent target training seeds is reported separately; MB adaptation checkpoints are a distinct repeat type.3×10 is not30 independent trained policies. Raw same-seed deltas live in PAIRED_RETURN_DELTAS.csv. Every step reward/privileged critic input/termination/raw artifact is retained, including checkpoint SHA and full readbacks. Pairing audit marks actual equality of initial states/config/terrain; it does not require later diverging trajectories to share PhysX states.', '', 'Frozen new train/validation/test/critic/anchor roles are disjoint and no primary evaluation preceded final policy selection. Historical source PPO/WM original seed records are unavailable; absolute historical-data non-overlap cannot be certified. Privileged actor/critic receive normalized mass delta (8 at target), normalized gains (0 at fixed source-reference gain), foot wrenches/heights, material and joint-friction fields. Full field dimensions and actual value ranges are retained in runtime readbacks. The SimDist latent value input and its controlled policy differ; no critic weight transplant or oracle claim is made.', '', 'Historical anchors remain M0=36.259520,MB1=38.904130,MB2=37.812854,MB3=35.935677 with ET4/5,4/5,4/5,5/5. They are not added to the fresh primary sample. Source PPO0kg versus M00kg measures an information/controller/optimization mixture; any difference-of-gaps is descriptive only. No f/V/R intervention, new WM adaptation, T1/T2/T3, or other gap screening occurred.', '', 'Seed62001 resumed once from target iteration250 after a software inference-buffer read failure. Actor/critic/Adam/Python/NumPy/Torch RNG and target counters were restored, while fixed same-seed PhysX environment state was reconstructed. An intermediate loader attempt failed on an unused missing infos metadata key before any training step; original checkpoint was preserved and direct state loading repaired the interface. Internal contact/sensor/command histories are not exact-replayed. The same inference-read fix applies to all three seeds; physics/reward/observation/action and PPO hyperparameters are unchanged. Failure and resume logs, all validation checkpoints and validation-best/final selection remain available.', '', 'Canonical Go2SimEnvCfg disables actor observation corruption in target training and evaluation, while the source training recipe declares Gaussian observation noises. This fixed-contract change is explicit. Source runner cohort boundaries also restart episode-log bookkeeping; train episodic logs may contain partial sums there, unlike fixed1000-step validation/primary Return. Loop wall time contains simulator/logging overhead and does not establish a matched controller deployment cost;50Hz is the simulated control rate.', '', '![Performance](../07_summary/PERFORMANCE_GAP_PLOT.png)', '', '![Validation](../07_summary/PPO_VALIDATION_LEARNING_CURVES.png)']
lines += ['', '## Additional preserved failure and recovery evidence', '', 'Seed62002 originally completed1500 training iterations but its co-resident validation initialization exhausted GPU memory before any rollout. The failed validation directory/log/checkpoint were preserved. Recovery temporarily offloads completed rollout storage while validation runs, retaining the persistent training environment; settings and frozen selection rule stay fixed. CPU selection-rule replay and checkpoint checks are recorded separately from subsequent GPU execution. Process resumes reconstruct PhysX state and do not establish exact replay.', '', '| seed | resume checkpoint | resumed UTC | restored contract |', '|---:|---|---|---|']
for seed in [62001, 62002, 62003]:
    for p in sorted((ROOT / '02_training' / f'seed_{seed}').glob('RESUME_[0-9]*.json')):
        event = read(p)
        lines.append(f"|{seed}|`{event['checkpoint']}`|{event['resumed_utc']}|{event['restored']}|")
report = '\n'.join(lines) + '\n'
(ROOT / '08_report/MASS8_GAP_COMPARISON.md').write_text(report)
(ROOT / '02_training/PPO_CONVERGENCE_AUDIT.md').write_text('\n'.join(lines[lines.index('## Convergence: each independent training seed'):lines.index('## Critic support diagnostics')]) + '\n')
with (ROOT / '02_training/PPO_CONVERGENCE_AUDIT.md').open('a') as f:
    f.write('\n' + '\n'.join(lines[lines.index('## Critic support diagnostics'):lines.index('## Historical corrected S4 consistency, separate provenance')]) + '\n')
manifest = read(ROOT / '09_provenance/RUN_MANIFEST.json')
manifest.update(status='VERIFYING_FINAL_EVIDENCE', primary_results_seen=True, report=str(ROOT / '08_report/MASS8_GAP_COMPARISON.md'), training_seeds_completed=3, primary_rollouts_completed=70, anchor_rollouts_completed=20, critic_rollouts_completed=30)
(ROOT / '09_provenance/RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
verified = subprocess.run([sys.executable, str(PORTFOLIO_ROOT / 'analysis/privileged_ppo/final_verify.py')], check=False)
manifest.update(status='COMPLETE' if verified.returncode == 0 else 'FINAL_VERIFICATION_FAILED', completed_utc=datetime.now(timezone.utc).isoformat())
(ROOT / '09_provenance/RUN_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
assert verified.returncode == 0, 'Final verification failed; inspect retained raw evidence and verification artifact'
hashes = {str(p): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size} for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name != 'FILE_HASHES.json' and ('__pycache__' not in p.parts)}
(ROOT / '09_provenance/FILE_HASHES.json').write_text(json.dumps({'scope': 'All new-root files except this self-referential inventory and Python bytecode', 'files': hashes}, indent=2) + '\n')
print('REPORT_COMPLETE', ROOT / '08_report/MASS8_GAP_COMPARISON.md')
