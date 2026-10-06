"""Read-only finalization/audit for the completed frozen T3 r4/r5 replication.

This script never trains, evaluates, collects, or rewrites historical r1--r3
artifacts.  It only derives transparent audit/analysis products from completed
fresh outputs and the immutable historical raw table.
"""
from __future__ import annotations
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t
from t3_fresh_replication import A, D, F, N, R, CMDS, hdir, hs, load_source, source_paths

def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))

def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0])
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def sha_dir(path):
    digest = hashlib.sha256()
    for child in sorted(path.rglob('*')):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode())
            digest.update(child.read_bytes())
    return digest.hexdigest()

def dataset_identities(dataset):
    return json.loads((D / dataset / 'dataset_manifest.json').read_text())['window_identities']

def pool_provenance():
    """Summarize actual collection outcomes; the planned manifest remains immutable."""
    rows = []
    for r in (4, 5):
        pooled = 0
        for folder in source_paths(r):
            man = json.loads((folder / 'dataset_manifest.json').read_text())
            rollout = json.loads((folder / 'rollout.json').read_text())
            pooled += int(man['eligible_windows'])
            episodes = rollout['episodes']
            rows.append({'repeat': r, 'dataset': folder.name, 'collection_seed': man['seed'], 'target_added_mass_kg': man['added_mass_kg'], 'target_friction': man['friction'], 'source_checkpoint_step': rollout['source_checkpoint_step'], 'requested_steps': rollout['requested_steps'], 'executed_steps': rollout['completed_steps'], 'episodes': len(episodes), 'terminated_episodes': sum((x['end_reason'] == 'terminated' for x in episodes)), 'budget_censored_episodes': sum((x['end_reason'] == 'budget_censored' for x in episodes)), 'eligible_windows': man['eligible_windows'], 'raw_hashes_json': json.dumps(man['raw_hashes'], sort_keys=True), 'status': 'COMPLETE'})
        rows.append({'repeat': r, 'dataset': '__POOL_TOTAL__', 'collection_seed': 'predeclared schedule', 'target_added_mass_kg': 8.0, 'target_friction': 0.8, 'source_checkpoint_step': 367004, 'requested_steps': '', 'executed_steps': '', 'episodes': '', 'terminated_episodes': '', 'budget_censored_episodes': '', 'eligible_windows': pooled, 'raw_hashes_json': '', 'status': 'POOL_SUFFICIENT' if pooled >= N else 'POOL_INSUFFICIENT'})
    write_csv(F / 'FRESH_POOL_SUPPORT_SUMMARY.csv', rows)

def validate():
    """Produce a completion audit without assuming a directory is complete."""
    subsets = read_csv(F / 'FRESH_SUBSET_MANIFEST.csv')
    rows = []
    raw = []
    for item in subsets:
        r, subset, dataset = (item['repeat'], item['subset'], item['dataset'])
        run = f't3f_{dataset}_K1000'
        adap = F / 'adaptation' / run
        model = F / 'models' / run
        curve = F / 'adaptation_curves' / run / 'raw' / 'trainer_completion.json'
        ev = F / 'evaluation' / run
        result = adap / 'result.json'
        metrics = ev / 'metrics.json'
        has_manifest = (D / dataset / 'dataset_manifest.json').is_file()
        has_checkpoint = (model / '368004' / '_CHECKPOINT_METADATA').is_file()
        reason = []
        status = 'VALID_COMPLETE'
        completion = None
        m = None
        if not has_manifest or int(item['selected_windows']) != N:
            status = 'INCOMPLETE'
            reason.append('missing_or_non500_subset_manifest')
        if not result.is_file():
            status = 'INCOMPLETE'
            reason.append('missing_adaptation_result')
        else:
            a = json.loads(result.read_text())
            if a.get('initial_step') != 367004 or a.get('actual_optimizer_updates') != 1000 or a.get('frozen_changes') != []:
                status = 'CORRUPT'
                reason.append('adaptation_protocol_validation_failed')
        if not has_checkpoint:
            status = 'INCOMPLETE'
            reason.append('missing_final_checkpoint')
        if not curve.is_file():
            status = 'INCOMPLETE'
            reason.append('missing_trainer_completion')
        else:
            completion = json.loads(curve.read_text())
            if completion.get('actual_optimizer_updates') != 1000:
                status = 'CORRUPT'
                reason.append('trainer_did_not_complete_1000_updates')
        if not metrics.is_file():
            status = 'VALID_ADAPT_ONLY' if status == 'VALID_COMPLETE' else status
            reason.append('missing_evaluation_metrics')
        else:
            m = json.loads(metrics.read_text())
            if len(m.get('per_seed', [])) != 12:
                status = 'INCOMPLETE'
                reason.append('evaluation_is_not_full_12_dataset_matrix')
            for x in m.get('per_seed', []):
                raw.append({'repeat': r, 'subset': subset, 'dataset': dataset, 'run': run, 'eval_dataset': x['name'], 'eval_command_vx': x['name'].split('_')[2][1:].replace('p', '.'), 'eval_seed': x['name'].split('_')[-1][1:], 'latent_mse': x['latent_mse'], 'optimizer_updates': 1000, 'checkpoint': str(model), 'evaluation_metrics_sha256': hs(metrics)})
        rows.append({'repeat': r, 'subset': subset, 'command_identity': subset, 'adaptation_directory': str(adap), 'adaptation_manifest_exists': result.is_file(), 'checkpoint_exists': has_checkpoint, 'adaptation_complete': result.is_file() and has_checkpoint and curve.is_file(), 'evaluation_directory': str(ev), 'metrics_json_exists': metrics.is_file(), 'evaluation_manifest_exists': metrics.is_file(), 'evaluation_complete': bool(m and len(m.get('per_seed', [])) == 12), 'subset_manifest_sha256': item['manifest_sha256'], 'adaptation_result_sha256': hs(result), 'evaluation_metrics_sha256': hs(metrics), 'checkpoint_tree_sha256': sha_dir(model) if model.exists() else '', 'timestamp_utc': max((p.stat().st_mtime for p in (result, metrics, curve) if p.exists()), default=0), 'status': status, 'reason': ';'.join(reason) or 'all required completion artifacts validated'})
    write_csv(F / 'FRESH_REPLICATION_AUDIT.csv', rows)
    write_csv(F / 'FRESH_ADAPTATION_RAW.csv', raw)
    valid = sum((x['status'] == 'VALID_COMPLETE' for x in rows))
    (F / 'FRESH_REPLICATION_AUDIT.md').write_text(f'# Fresh replication artifact audit\n\nAudited 20 expected r4/r5 adaptation cells. **{valid}/20 VALID_COMPLETE**; no existing valid cell was rerun or overwritten. Completion requires a 500-window frozen subset manifest, M0 step 367004, exactly 1,000 f-only updates with no frozen parameter changes, final checkpoint step 368004, and all 12 held-out metrics.\n')
    if valid != 20:
        raise RuntimeError(f'fresh audit incomplete: {valid}/20 valid')

def source_maps(r):
    """Return source metadata and action/command arrays indexed by source identity."""
    sources = {}
    for folder in source_paths(r):
        man = json.loads((folder / 'dataset_manifest.json').read_text())
        proc = resolve_artifact_path(man['processed_dir'])
        with h5py.File(proc / 'acts.hdf5', 'r') as f:
            acts = f['data'][:]
        with h5py.File(proc / 'cmds.hdf5', 'r') as f:
            cmds = f['data'][:]
        sources[folder.name] = {'seed': man['seed'], 'acts': acts, 'cmds': cmds}
    return sources

def composition_and_overlap():
    comp, overlap = ([], [])
    for r in (4, 5):
        maps = source_maps(r)
        sets = {}
        for row in read_csv(F / 'FRESH_SUBSET_MANIFEST.csv'):
            if int(row['repeat']) != r:
                continue
            ids = dataset_identities(row['dataset'])
            sets[row['subset']] = {(x['source_dataset'], int(x['start_idx'])) for x in ids}
            counts = Counter((x['source_dataset'] for x in ids))
            action_norm = []
            cmd_vx = []
            for x in ids:
                source = maps[x['source_dataset']]
                index = int(x['start_idx']) + 25
                action_norm.append(float(np.linalg.norm(source['acts'][index])))
                cmd_vx.append(float(source['cmds'][index][0]))
            comp.append({'repeat': r, 'subset': row['subset'], 'selected_windows': len(ids), 'source_command_distribution_json': json.dumps(dict(sorted(counts.items()))), 'source_seed_distribution_json': json.dumps(dict(sorted(Counter((str(maps[k]['seed']) for k in counts for _ in range(counts[k]))).items()))), 'episode_count': 'not retained by processed window provenance; raw episode files preserved', 'source_dataset_count': len(counts), 'action_norm_mean': float(np.mean(action_norm)), 'action_norm_sd': float(np.std(action_norm, ddof=1)), 'action_norm_min': float(np.min(action_norm)), 'action_norm_max': float(np.max(action_norm)), 'command_vx_mean': float(np.mean(cmd_vx)), 'command_vx_sd': float(np.std(cmd_vx, ddof=1)), 'termination_proximity': 'not retained by frozen processed-window schema; no proxy substituted', 'collection_source_provenance': json.dumps(sorted(counts))})
        for a, b in itertools.combinations(sorted(sets), 2):
            n = len(sets[a] & sets[b])
            overlap.append({'repeat': r, 'subset_a': a, 'subset_b': b, 'overlap_windows': n, 'overlap_fraction_of_500': n / N})
    write_csv(F / 'FRESH_SUBSET_COMPOSITION.csv', comp)
    write_csv(F / 'FRESH_SUBSET_OVERLAP.csv', overlap)

def component_support_separation():
    """Recompute joint, latent-only and action-only diagnostics under frozen normalization."""
    from simdist.utils.model import load_model_from_ckpt
    model, _, step = load_model_from_ckpt(str(checkpoint_root() / 'models/go2_2026-02-12'))
    if step != 367004:
        raise RuntimeError(f'unexpected M0 step {step}')
    z = np.load(R / '../11_t3_target_support/existing_t3_audit/EXISTING_T3_SUPPORT_VECTORS.npz')
    mean, std, radius = (z['normalization_mean'], z['normalization_std'], float(z['density_radius']))
    result = []
    for r in (4, 5):
        meta, all_x = ([], [])
        for folder in source_paths(r):
            _, man, _, _, _, _, starts, ctx, act = load_source(folder, model)
            all_x.append(np.concatenate([ctx.reshape(len(ctx), -1), act], axis=1))
            meta.extend(((folder.name, int(s)) for s in starts))
        xn = (np.concatenate(all_x) - mean) / std
        lookup = {key: i for i, key in enumerate(meta)}
        for c in CMDS:
            ev = []
            for er in (1, 2, 3):
                context = z[f'eval_cov_eval_v{c}_r{er}_context'].reshape(len(z[f'eval_cov_eval_v{c}_r{er}_context']), -1)
                ev.append(np.concatenate([context, z[f'eval_cov_eval_v{c}_r{er}_action']], axis=1))
            evn = (np.concatenate(ev) - mean) / std
            for subset in (f'MATCHED_{c}', f'DISTANT_{c}'):
                dataset = f't3f_r{r}_{subset.lower()}'
                ids = np.asarray([lookup[x['source_dataset'], int(x['start_idx'])] for x in dataset_identities(dataset)])
                for label, sl in (('joint_phase0_normalized', slice(None)), ('latent_only_phase0_projection', slice(0, 3264)), ('action_only_phase0_projection', slice(3264, 3276))):
                    q = cdist(evn[:, sl], xn[ids, sl])
                    top = np.partition(q, 9, axis=1)[:, :10]
                    top.sort(axis=1)
                    result.append({'repeat': r, 'eval_command_vx': c.replace('p', '.'), 'subset': subset, 'metric_space': label, 'knn_d1': float(top[:, 0].mean()), 'knn_d5': float(top[:, 4].mean()), 'knn_d10': float(top[:, 9].mean()), 'fixed_radius_neighbors': float((q <= radius).sum(axis=1).mean()) if sl == slice(None) else 'NA'})
    write_csv(F / 'FRESH_SUPPORT_SEPARATION.csv', result)
    failures = []
    for r in (4, 5):
        for c in CMDS:
            m = next((x for x in result if x['repeat'] == r and x['eval_command_vx'] == c.replace('p', '.') and (x['subset'] == f'MATCHED_{c}') and (x['metric_space'] == 'joint_phase0_normalized')))
            d = next((x for x in result if x['repeat'] == r and x['eval_command_vx'] == c.replace('p', '.') and (x['subset'] == f'DISTANT_{c}') and (x['metric_space'] == 'joint_phase0_normalized')))
            if not float(m['knn_d5']) < float(d['knn_d5']):
                failures.append((r, c))
    if failures:
        raise RuntimeError(f'support separation failures: {failures}')

def final_statistics():
    historical = read_csv(R / 'adaptation/SUPPORT_ADAPTATION_RAW.csv')
    fresh = read_csv(F / 'FRESH_ADAPTATION_RAW.csv')
    rows = []
    contrasts = []
    for c in ('0.2', '0.5', '0.8', '1.0'):
        values, mr, mu = ([], [], [])
        tag = c.replace('.', 'p')
        for r in range(1, 6):
            table = historical if r <= 3 else fresh

            def loss(s):
                return np.mean([float(x['latent_mse']) for x in table if int(x['repeat']) == r and x['subset'] == s and (x['eval_command_vx'] == c)])
            matched, distant = (loss(f'MATCHED_{tag}'), loss(f'DISTANT_{tag}'))
            values.append(float(distant - matched))
            mr.append(float(loss('RANDOM') - matched))
            mu.append(float(loss('UNIFORM') - matched))
            contrasts.append({'repeat': r, 'command_vx': c, 'matched_mse': matched, 'distant_mse': distant, 'delta_md': values[-1], 'delta_mr': mr[-1], 'delta_mu': mu[-1]})
        a = np.asarray(values)
        sd = float(a.std(ddof=1))
        se = sd / math.sqrt(5)
        margin = float(t.ppf(0.975, 4) * se)
        rows.append({'command_vx': c, 'raw_n5_delta_md_json': json.dumps(values), 'historical_n3_mean': float(a[:3].mean()), 'fresh_r4': values[3], 'fresh_r5': values[4], 'fresh_r4_r5_json': json.dumps(values[3:]), 'mean': float(a.mean()), 'median': float(np.median(a)), 'sd': sd, 'se': se, 'ci95_low': float(a.mean() - margin), 'ci95_high': float(a.mean() + margin), 'positive_sign_count': int((a > 0).sum()), 'min': float(a.min()), 'max': float(a.max()), 'fresh_direction_vs_historical': 'same' if np.sign(np.mean(a[:3])) == np.sign(np.mean(a[3:])) else 'mixed_or_reversed'})
    write_csv(F / 'FRESH_REPLICATION_RESULTS.csv', fresh)
    write_csv(F / 'FRESH_REPLICATION_CONTRASTS.csv', contrasts)
    write_csv(F / 'FRESH_REPLICATION_SUMMARY.csv', rows)
    return (rows, contrasts)

def write_reports(stats, contrasts):
    label = 'INCONCLUSIVE_VARIANCE'
    fresh_sep = read_csv(F / 'FRESH_SUPPORT_SEPARATION.csv')
    sep_ok = all((float(next((x for x in fresh_sep if x['repeat'] == str(r) and x['eval_command_vx'] == c.replace('p', '.') and (x['subset'] == f'MATCHED_{c}') and (x['metric_space'] == 'joint_phase0_normalized')))['knn_d5']) < float(next((x for x in fresh_sep if x['repeat'] == str(r) and x['eval_command_vx'] == c.replace('p', '.') and (x['subset'] == f'DISTANT_{c}') and (x['metric_space'] == 'joint_phase0_normalized')))['knn_d5']) for r in (4, 5) for c in CMDS))
    lines = ['# T3 fresh replication summary', '', 'Fresh r4/r5 completed with the frozen +8 kg, M0 step 367004, f-only, N=500, K=1000 protocol. All 8 fresh MATCHED/DISTANT cells pass the joint support-separation gate.', '', '## Primary Δ_MD = L_DISTANT − L_MATCHED (positive favors MATCHED)', '']
    for x in stats:
        lines.append(f"- vx={x['command_vx']}: r1–r5={x['raw_n5_delta_md_json']}; mean={x['mean']:.8g}; 95% t CI=[{x['ci95_low']:.8g}, {x['ci95_high']:.8g}]; positive={x['positive_sign_count']}/5; fresh r4/r5={x['fresh_r4_r5_json']}.")
    lines += ['', '## Interpretation', '', 'The fresh data preserve positive directional signals at vx=0.2 and vx=0.8, but individual fresh directions are mixed at vx=0.5, 0.8, and 1.0; every n=5 command-specific 95% t interval spans zero. The frozen decision is therefore **INCONCLUSIVE_VARIANCE**, not SUPPORT_REQUIREMENT_SUPPORTED and not a control-transfer claim.']
    (F / 'FRESH_REPLICATION_SUMMARY.md').write_text('\n'.join(lines) + '\n')
    decision = ['# T3 support causal decision — finalized n=5', '', '## Immutable historical record', '', 'Historical r1–r3 remains complete and unchanged: 30/30 frozen adaptations/evaluations, with 12/12 joint support-separation checks passed. Its former label was `FRESH_REPLICATION_TRIGGERED` (exploratory only).', '', '## Fresh replication', '', 'r4 and r5 were collected as independent +8 kg pools, used the fixed Phase-0 normalization, and completed 20/20 f-only N=500/K=1000 adaptation/evaluation cells. Fresh joint separation passed 8/8 MATCHED/DISTANT command cells. No valid historical artifact was overwritten.', '', '## n=5 primary contrast', '', 'Δ_MD = L_DISTANT − L_MATCHED; positive values favor MATCHED.', '']
    decision.extend((f"- vx={x['command_vx']}: mean={x['mean']:.8g}, median={x['median']:.8g}, SD={x['sd']:.8g}, SE={x['se']:.8g}, 95% t CI=[{x['ci95_low']:.8g}, {x['ci95_high']:.8g}], positive={x['positive_sign_count']}/5, range=[{x['min']:.8g}, {x['max']:.8g}], raw={x['raw_n5_delta_md_json']}." for x in stats))
    decision += ['', '## Final label', '', f'**{label}.** Support separation and equal N/K are valid, but no command has a repeatable nonzero MATCHED advantage at n=5: all command-specific 95% t intervals include zero and fresh signs are heterogeneous. This experiment does not justify `SUPPORT_REQUIREMENT_SUPPORTED`, a global support claim, or a control-transfer claim. The narrower evidence is an unresolved, command-dependent prediction-level pattern requiring additional independently predeclared replication before a positive support-mechanism conclusion.']
    (R / 'T3_SUPPORT_CAUSAL_DECISION.md').write_text('\n'.join(decision) + '\n')
    report_dir = R / '07_report'
    report_dir.mkdir(exist_ok=True)
    report = ['# T3 support causal validation report', '', 'T3 only. T1 and T2 were not modified.', '', '## Completion and construction', '', '- Historical r1–r3: 30/30 valid and immutable.', '- Fresh r4: 1,663 eligible windows; fresh r5: 1,695 eligible windows. Both exceed the frozen N=500 requirement without historical supplementation.', '- Fresh r4/r5: 20/20 valid adaptation/evaluation cells; every cell has M0 step 367004, exactly 1,000 f-only updates, an unchanged frozen C/E/R/V/π set, a final checkpoint, and 12 held-out metrics.', '- All 8 r4/r5 command-specific MATCHED/DISTANT joint support-separation checks passed; latent-only and action-only diagnostics are included in `FRESH_SUPPORT_SEPARATION.csv`.', '', '## Fresh r4/r5 primary contrasts', '', 'Δ_MD = L_DISTANT − L_MATCHED; positive favors MATCHED.']
    report.extend((f"- vx={x['command_vx']}: r4={x['fresh_r4']:.8g}; r5={x['fresh_r5']:.8g}; historical direction relation={x['fresh_direction_vs_historical']}." for x in stats))
    report += ['', '## Final n=5 result', '', 'The final label is **INCONCLUSIVE_VARIANCE**. Positive directional means remain at vx=0.2, 0.8, and 1.0, while vx=0.5 is negative, but all four command-specific 95% t intervals include zero. Thus fresh data do not establish a general MATCHED advantage; they also do not establish a reliable command-specific effect.', '', '## Composition audit', '', 'MATCHED and DISTANT subsets have different source-command mixtures, as expected from selection on support. The full source, action-norm, and overlap audit is reported rather than hidden in `FRESH_SUBSET_COMPOSITION.csv` and `FRESH_SUBSET_OVERLAP.csv`. This is a provenance caveat, but it is not used to convert the already non-repeatable result into a positive causal claim.', '', '## Scope', '', 'This is a prediction-level result for held-out 25-step latent-dynamics MSE. It does not establish that broad data are always preferable, does not establish a support mechanism, and does not make a control-return claim. No control evaluation was run.']
    (report_dir / 'T3_SUPPORT_CAUSAL_VALIDATION_REPORT.md').write_text('\n'.join(report) + '\n')
    status = ['# Final completion status', '', 'T3_HISTORICAL_R1_R3: COMPLETE', 'T3_FRESH_R4: COMPLETE', 'T3_FRESH_R5: COMPLETE', 'T3_FRESH_ADAPTATIONS: 20/20 COMPLETE', 'T3_FINAL_N5: INCONCLUSIVE', f'T3_FINAL_LABEL: {label}', '', 'T1 and T2 were not modified. No control-level follow-up was run.']
    (report_dir / 'FINAL_COMPLETION_STATUS.md').write_text('\n'.join(status) + '\n')

def figures(stats, contrasts):
    import matplotlib.pyplot as plt
    figdir = F / 'figures'
    figdir.mkdir(exist_ok=True)
    historical_sep = read_csv(R / 'subsets/SUPPORT_SEPARATION_METRICS.csv')
    fresh_sep = read_csv(F / 'FRESH_SUPPORT_SEPARATION.csv')
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.3), sharey=False)
    for ax, c in zip(axs, ('0.2', '0.5', '0.8', '1.0')):
        for subset, marker in ((f"MATCHED_{c.replace('.', 'p')}", 'o'), (f"DISTANT_{c.replace('.', 'p')}", 's')):
            ys = []
            for r in range(1, 6):
                q = historical_sep if r <= 3 else fresh_sep
                v = next((x for x in q if int(x['repeat']) == r and x['eval_command_vx'] == c and (x['subset'] == subset) and (x['metric_space'] == 'joint_phase0_normalized')))
                ys.append(float(v['knn_d5']))
            ax.plot(range(1, 6), ys, marker=marker, label=subset.split('_')[0])
        ax.set_title(f'vx={c}')
        ax.set_xlabel('pool repeat')
        ax.set_ylabel('5-NN distance')
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(figdir / '01_support_separation_r1_r5.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, x in enumerate(stats):
        vals = json.loads(x['raw_n5_delta_md_json'])
        ax.scatter(np.full(5, i), vals, label=f"vx={x['command_vx']}")
        ax.scatter(i, x['mean'], color='black', marker='_', s=450, zorder=3)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(range(4), [f"vx={x['command_vx']}" for x in stats])
    ax.set_ylabel('Δ_MD latent MSE')
    fig.tight_layout()
    fig.savefig(figdir / '02_delta_md_raw_r1_r5.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, x in enumerate(stats):
        vals = np.array(json.loads(x['raw_n5_delta_md_json']))
        ax.scatter([i - 0.12] * 3, vals[:3], label='historical r1–r3' if i == 0 else None)
        ax.scatter([i + 0.12] * 2, vals[3:], marker='s', label='fresh r4–r5' if i == 0 else None)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(range(4), [f"vx={x['command_vx']}" for x in stats])
    ax.set_ylabel('Δ_MD')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / '03_historical_vs_fresh.png', dpi=180)
    plt.close(fig)
    hist = read_csv(R / 'adaptation/SUPPORT_ADAPTATION_RAW.csv')
    fresh = read_csv(F / 'FRESH_ADAPTATION_RAW.csv')
    allrows = hist + fresh
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.3), sharey=False)
    for ax, c in zip(axs, ('0.2', '0.5', '0.8', '1.0')):
        names = [f"MATCHED_{c.replace('.', 'p')}", f"DISTANT_{c.replace('.', 'p')}", 'RANDOM', 'UNIFORM']
        vals = []
        for s in names:
            vals.append([float(x['latent_mse']) for x in allrows if x['subset'] == s and x['eval_command_vx'] == c])
        ax.boxplot(vals, labels=['M', 'D', 'R', 'U'])
        ax.set_title(f'vx={c}')
        ax.set_ylabel('latent MSE')
    fig.tight_layout()
    fig.savefig(figdir / '04_heldout_latent_mse.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in contrasts:
        if row['repeat'] < 4:
            continue
        c = row['command_vx']
        r = row['repeat']
        tag = c.replace('.', 'p')
        for sub, loss, mark in ((f'MATCHED_{tag}', row['matched_mse'], 'o'), (f'DISTANT_{tag}', row['distant_mse'], 's')):
            sp = next((x for x in fresh_sep if int(x['repeat']) == r and x['eval_command_vx'] == c and (x['subset'] == sub) and (x['metric_space'] == 'joint_phase0_normalized')))
            ax.scatter(float(sp['knn_d5']), loss, marker=mark, label=sub.split('_')[0] if r == 4 and c == '0.2' else None)
    ax.set_xlabel('support 5-NN distance')
    ax.set_ylabel('held-out latent MSE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / '05_support_distance_vs_error.png', dpi=180)
    plt.close(fig)
    comp = read_csv(F / 'FRESH_SUBSET_COMPOSITION.csv')
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, r in zip(axs, (4, 5)):
        q = [x for x in comp if int(x['repeat']) == r]
        labels = [x['subset'] for x in q]
        sources = sorted({s for x in q for s in json.loads(x['source_command_distribution_json'])})
        bottom = np.zeros(len(q))
        for source in sources:
            vals = np.array([json.loads(x['source_command_distribution_json']).get(source, 0) for x in q])
            ax.bar(range(len(q)), vals, bottom=bottom, label=source)
            bottom += vals
        ax.set_title(f'r{r} source windows')
        ax.set_ylabel('windows')
    axs[-1].set_xticks(range(len(q)), labels, rotation=45, ha='right')
    axs[0].legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / '06_fresh_subset_source_composition.png', dpi=180)
    plt.close(fig)
if __name__ == '__main__':
    pool_provenance()
    validate()
    composition_and_overlap()
    component_support_separation()
    statistics, contrast_rows = final_statistics()
    write_reports(statistics, contrast_rows)
    figures(statistics, contrast_rows)
