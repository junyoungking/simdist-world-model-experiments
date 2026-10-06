"""Summarize completed frozen T3 cells without changing any experimental data."""
from __future__ import annotations
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t
WORK = PORTFOLIO_ROOT
ROOT = legacy_dir('11_t3_support_causal')
OLD = legacy_dir('11_t3_target_support') / 'existing_t3_audit'
CONDS = ('MATCHED_0p2', 'DISTANT_0p2', 'MATCHED_0p5', 'DISTANT_0p5', 'MATCHED_0p8', 'DISTANT_0p8', 'MATCHED_1p0', 'DISTANT_1p0', 'UNIFORM', 'RANDOM')
CMDS = ('0.2', '0.5', '0.8', '1.0')

def wc(path, rows, fields=None):
    fields = fields or list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def support(evalx, sub, radius):
    d = cdist(evalx, sub)
    p = np.partition(d, 9, axis=1)[:, :10]
    p.sort(axis=1)
    return dict(knn_d1=float(p[:, 0].mean()), knn_d5=float(p[:, 4].mean()), knn_d10=float(p[:, 9].mean()), fixed_radius_neighbors=float((d <= radius).sum(axis=1).mean()))

def main():
    raw = list(csv.DictReader((ROOT / 'adaptation/SUPPORT_ADAPTATION_RAW.csv').open()))
    if len(raw) != 360:
        raise RuntimeError(f'Need 360 frozen evaluation rows, got {len(raw)}')
    by = defaultdict(list)
    for x in raw:
        by[int(x['repeat']), x['subset'], x['eval_command_vx']].append(float(x['latent_mse']))
    summary = []
    for r in (1, 2, 3):
        for s in CONDS:
            for c in CMDS:
                v = by[r, s, c]
                if len(v) != 3:
                    raise RuntimeError((r, s, c, len(v)))
                summary.append({'repeat': r, 'subset': s, 'eval_command_vx': c, 'heldout_seed_count': len(v), 'latent_mse_mean': float(np.mean(v)), 'latent_mse_sd': float(np.std(v, ddof=1)), 'latent_mse_values_json': json.dumps(v)})
    wc(ROOT / 'adaptation/SUPPORT_ADAPTATION_REPEAT_SUMMARY.csv', summary)
    contrast = []
    for c in CMDS:
        vals = {k: [] for k in ('MD', 'MR', 'MU')}
        for r in (1, 2, 3):
            g = {s: float(np.mean(by[r, s, c])) for s in CONDS}
            tag = c.replace('.', 'p')
            vals['MD'].append(g['DISTANT_' + tag] - g['MATCHED_' + tag])
            vals['MR'].append(g['RANDOM'] - g['MATCHED_' + tag])
            vals['MU'].append(g['UNIFORM'] - g['MATCHED_' + tag])
        for name, v in vals.items():
            a = np.asarray(v)
            sd = float(np.std(a, ddof=1))
            se = sd / math.sqrt(3)
            margin = float(t.ppf(0.975, 2) * se)
            contrast.append({'command_vx': c, 'contrast': name, 'raw_repeat_values_json': json.dumps(v), 'mean': float(a.mean()), 'median': float(np.median(a)), 'sd': sd, 'se': se, 'ci95_low': float(a.mean() - margin), 'ci95_high': float(a.mean() + margin), 'positive_sign_count': int((a > 0).sum()), 'min': float(a.min()), 'max': float(a.max())})
    wc(ROOT / 'T3_PRIMARY_CAUSAL_CONTRASTS.csv', contrast)
    prior = list(csv.DictReader((OLD / 'EXISTING_T3_SUPPORT_RAW.csv').open()))
    adap = [x for x in prior if x['row_kind'] == 'adaptation']
    z = np.load(OLD / 'EXISTING_T3_SUPPORT_VECTORS.npz')
    mean, std, radius = (z['normalization_mean'], z['normalization_std'], float(z['density_radius']))
    sep = []
    for r in (1, 2, 3):
        meta = []
        joint = []
        latent = []
        action = []
        for cond in ('NARROW', 'BROAD'):
            m = [x for x in adap if x['condition'] == cond and int(x['repeat']) == r]
            ctx = z[f'adapt_{cond}_r{r}_context']
            act = z[f'adapt_{cond}_r{r}_action']
            assert len(m) == len(ctx)
            meta += m
            joint.append(np.concatenate([ctx.reshape(len(ctx), -1), act], axis=1))
            latent.append(ctx.reshape(len(ctx), -1))
            action.append(act)
        joint = np.concatenate(joint)
        latent = np.concatenate(latent)
        action = np.concatenate(action)
        index = {(m['dataset'], str(m['start_idx'])): i for i, m in enumerate(meta)}
        for c in CMDS:
            cc = c.replace('.', 'p')
            e_joint = []
            e_lat = []
            e_act = []
            for er in (1, 2, 3):
                ctx = z[f'eval_cov_eval_v{cc}_r{er}_context']
                act = z[f'eval_cov_eval_v{cc}_r{er}_action']
                e_joint.append(np.concatenate([ctx.reshape(len(ctx), -1), act], axis=1))
                e_lat.append(ctx.reshape(len(ctx), -1))
                e_act.append(act)
            ej = np.concatenate(e_joint)
            el = np.concatenate(e_lat)
            ea = np.concatenate(e_act)
            for subset in (f'MATCHED_{cc}', f'DISTANT_{cc}'):
                man = json.loads((ROOT / 'datasets' / f't3c_r{r}_{subset.lower()}' / 'dataset_manifest.json').read_text())
                ids = [index[x['dataset'], str(x['start_idx'])] for x in man['window_identities']]
                spaces = (('joint_phase0_normalized', (ej - mean) / std, (joint[ids] - mean) / std), ('latent_only_phase0_projection', (el - mean[:el.shape[1]]) / std[:el.shape[1]], (latent[ids] - mean[:el.shape[1]]) / std[:el.shape[1]]), ('action_only_phase0_projection', (ea - mean[-ea.shape[1]:]) / std[-ea.shape[1]:], (action[ids] - mean[-ea.shape[1]:]) / std[-ea.shape[1]:]))
                for metric_space, ev, sub in spaces:
                    sep.append({'repeat': r, 'eval_command_vx': c, 'subset': subset, 'metric_space': metric_space, **support(ev, sub, radius)})
    wc(ROOT / 'subsets/SUPPORT_SEPARATION_METRICS.csv', sep)
    checks = []
    for r in (1, 2, 3):
        for c in CMDS:
            d = {x['subset']: x for x in sep if x['repeat'] == r and x['eval_command_vx'] == c and (x['metric_space'] == 'joint_phase0_normalized')}
            tag = c.replace('.', 'p')
            checks.append(d['MATCHED_' + tag]['knn_d5'] < d['DISTANT_' + tag]['knn_d5'])
    trigger = [x for x in contrast if x['contrast'] == 'MD' and x['mean'] > 0]
    lines = ['# T3 support causal decision', '', f'Historical frozen matrix: **30/30** adaptation/evaluation cells complete.', '', f'Joint-support separation (MATCHED d5 < DISTANT d5): **{sum(checks)}/{len(checks)}** repeat-command cells.', '', '## Exploratory primary contrast', '']
    for x in [x for x in contrast if x['contrast'] == 'MD']:
        lines.append(f"- vx={x['command_vx']}: mean Δ_MD={x['mean']:.8g}; positive repeats={x['positive_sign_count']}/3.")
    lines += ['', '## Label', '', '**FRESH_REPLICATION_TRIGGERED (exploratory only).** At least one predeclared command has positive mean Δ_MD (DISTANT MSE − MATCHED MSE). Historical n=3 does not establish SUPPORT_REQUIREMENT_SUPPORTED.' if trigger else '**NO_FRESH_REPLICATION_TRIGGER.** No predeclared command has a positive mean exploratory Δ_MD.', '']
    (ROOT / 'T3_SUPPORT_CAUSAL_DECISION.md').write_text('\n'.join(lines))
    print('T3_FINALIZE_COMPLETE', len(summary), len(contrast), sum(checks), len(checks), flush=True)
if __name__ == '__main__':
    main()
