"""Freeze T3 causal subset manifests and materialize equal-window dataset views.

This script deliberately never loads adapted checkpoints or prediction outcomes.
"""
from __future__ import annotations
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import csv, hashlib, json, shutil
from itertools import combinations
from pathlib import Path
import h5py, numpy as np
from scipy.spatial.distance import cdist
WORK = PORTFOLIO_ROOT
OLD = legacy_dir('11_t3_target_support') / 'existing_t3_audit'
CAUSAL = legacy_dir('11_t3_support_causal')
PRIOR = legacy_dir('research_topic_validation_20260914T012121Z_continuation')
N, H = (500, 25)
KEYS = ['proprio_obs.hdf5', 'extero_obs.hdf5', 'acts.hdf5', 'cmds.hdf5', 'rewards.hdf5', 'values.hdf5', 'expert_policy_flag.hdf5']
RSEEDS = {1: 95101, 2: 95102, 3: 95103}

def hashes(d):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob('*')) if p.is_file()}

def wc(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def q(a):
    return {f'q{x:02d}': float(np.quantile(a, x / 100)) for x in (5, 50, 95)}

def fps(x, identities):
    first = min(range(len(x)), key=lambda i: (identities[i]['dataset'], int(identities[i]['start_idx'])))
    chosen = [first]
    best = np.sum((x - x[first]) ** 2, axis=1)
    for _ in range(1, N):
        j = int(np.argmax(best))
        chosen.append(j)
        best = np.minimum(best, np.sum((x - x[j]) ** 2, axis=1))
    return np.asarray(chosen)

def write_view(name, selected, meta):
    out = CAUSAL / 'datasets' / name
    proc = out / 'processed_data_go2_H-25_T-25'
    out.mkdir(parents=True)
    proc.mkdir()
    sources = []
    offset = 0
    for source in sorted({m['dataset'] for m in meta}):
        folder = PRIOR / 'datasets' / source
        man = json.loads((folder / 'dataset_manifest.json').read_text())
        p = resolve_artifact_path(man['processed_dir'])
        with h5py.File(p / 'proprio_obs.hdf5', 'r') as f:
            length = len(f['data'])
        sources.append((source, p, offset, man))
        offset += length
    for key in KEYS:
        with h5py.File(proc / key, 'w') as dst:
            for _, p, _, _ in sources:
                with h5py.File(p / key, 'r') as src:
                    a = src['data'][:]
                    if 'data' not in dst:
                        ds = dst.create_dataset('data', shape=(0,) + a.shape[1:], maxshape=(None,) + a.shape[1:], dtype=a.dtype)
                    ds.resize(len(ds) + len(a), axis=0)
                    ds[-len(a):] = a
    starts = []
    identities = []
    offset_by = {s: o for s, _, o, _ in sources}
    for idx in selected:
        m = dict(meta[int(idx)])
        m['view_start_idx'] = int(m['start_idx']) + offset_by[m['dataset']]
        starts.append(m['view_start_idx'])
        identities.append(m)
    with h5py.File(proc / 'start_idxs.hdf5', 'w') as f:
        f.create_dataset('data', data=np.asarray(starts, dtype=np.int64))
    shutil.copy2(sources[0][1] / 'scaler_params.json', proc / 'scaler_params.json')
    shutil.copy2(sources[0][1] / 'system.yaml', proc / 'system.yaml')
    shutil.copy2(sources[0][1] / 'dataset_metrics.json', proc / 'dataset_metrics.json')
    man = {'name': name, 'role': 'collection', 'view_kind': 't3_causal_equal_size_subset', 'eligible_windows': N, 'seeds': sorted({z for _, _, _, m in sources for z in m['seeds']}), 'processed_dir': str(proc), 'processed_hashes': hashes(proc), 'window_identities': identities, 'source_datasets': [s for s, _, _, _ in sources]}
    (out / 'dataset_manifest.json').write_text(json.dumps(man, indent=2) + '\n')
    return man

def support(evalx, subset, radius):
    d = cdist(evalx, subset)
    part = np.partition(d, 9, axis=1)[:, :10]
    part.sort(axis=1)
    return {'d1': float(part[:, 0].mean()), 'd5': float(part[:, 4].mean()), 'd10': float(part[:, 9].mean()), 'density': float((d <= radius).sum(axis=1).mean())}

def main():
    if (CAUSAL / 'subsets/SUPPORT_SUBSET_MANIFEST.csv').exists():
        raise RuntimeError('subsets already frozen')
    raw = list(csv.DictReader((OLD / 'EXISTING_T3_SUPPORT_RAW.csv').open()))
    adap = [r for r in raw if r['row_kind'] == 'adaptation']
    z = np.load(OLD / 'EXISTING_T3_SUPPORT_VECTORS.npz')
    mean, std, radius = (z['normalization_mean'], z['normalization_std'], float(z['density_radius']))
    manifests = []
    comps = []
    seps = []
    allsets = {}
    for r in (1, 2, 3):
        meta = []
        xs = []
        lat = []
        act = []
        for cond in ('NARROW', 'BROAD'):
            m = [x for x in adap if x['condition'] == cond and int(x['repeat']) == r]
            context = z[f'adapt_{cond}_r{r}_context']
            a = z[f'adapt_{cond}_r{r}_action']
            x = np.concatenate([context.reshape(len(context), -1), a], axis=1)
            assert len(m) == len(x)
            meta += m
            xs.append(x)
            lat.append(context[:, -1, :])
            act.append(a)
        x = np.concatenate(xs)
        xn = (x - mean) / std
        lat = np.concatenate(lat)
        act = np.concatenate(act)
        sets = {}
        sets['UNIFORM'] = fps(xn, meta)
        sets['RANDOM'] = np.random.default_rng(RSEEDS[r]).choice(len(xn), N, replace=False)
        evals = {}
        for cc in ('0p2', '0p5', '0p8', '1p0'):
            e = []
            for er in (1, 2, 3):
                co = z[f'eval_cov_eval_v{cc}_r{er}_context']
                aa = z[f'eval_cov_eval_v{cc}_r{er}_action']
                e.append(np.concatenate([co.reshape(len(co), -1), aa], axis=1))
            ev = np.concatenate(e)
            evn = (ev - mean) / std
            evals[cc] = evn
            d = []
            for lo in range(0, len(xn), 64):
                ds = cdist(xn[lo:lo + 64], evn)
                d.extend(np.partition(ds, 4, axis=1)[:, :5].mean(axis=1))
            order = np.argsort(np.asarray(d), kind='stable')
            sets[f'MATCHED_{cc}'] = order[:N]
            sets[f'DISTANT_{cc}'] = order[-N:]
        allsets[r] = sets
        for label, idx in sets.items():
            man = write_view(f't3c_r{r}_{label.lower()}', idx, meta)
            digest = hashlib.sha256((CAUSAL / 'datasets' / man['name'] / 'dataset_manifest.json').read_bytes()).hexdigest()
            manifests.append({'repeat': r, 'subset': label, 'dataset': man['name'], 'selected_windows': N, 'manifest_sha256': digest})
            chosen = [meta[i] for i in idx]
            cmds = {}
            conds = {}
            eps = set()
            for m in chosen:
                cmds[m['command_vx']] = cmds.get(m['command_vx'], 0) + 1
                conds[m['condition']] = conds.get(m['condition'], 0) + 1
                eps.add((m['dataset'], m['episode_id']))
            norms = np.asarray([np.linalg.norm(np.asarray(json.loads(m['a_t']))) for m in chosen])
            comps.append({'repeat': r, 'subset': label, 'n': N, 'source_commands_json': json.dumps(cmds, sort_keys=True), 'source_conditions_json': json.dumps(conds, sort_keys=True), 'episode_count': len(eps), 'source_dataset_count': len({m['dataset'] for m in chosen}), 'termination_status_json': json.dumps({k: sum((m['termination_status'] == k for m in chosen)) for k in sorted({m['termination_status'] for m in chosen})}), 'action_norm_mean': float(norms.mean()), **q(norms)})
        for cc, evn in evals.items():
            for label, idx in sets.items():
                primary = support(evn, xn[idx], radius)
                ld = support(np.concatenate([z[f'eval_cov_eval_v{cc}_r{er}_latent'] for er in (1, 2, 3)]), (lat[idx] - lat.mean(0)) / np.maximum(lat.std(0), 1e-06), radius) if False else {}
                seps.append({'repeat': r, 'eval_command_vx': cc.replace('p', '.'), 'subset': label, **primary})
    for r, sets in allsets.items():
        for a, b in combinations(sorted(sets), 2):
            aa = set(map(int, sets[a]))
            bb = set(map(int, sets[b]))
            comps.append({'repeat': r, 'subset': f'OVERLAP::{a}::{b}', 'n': len(aa & bb), 'source_commands_json': '', 'source_conditions_json': '', 'episode_count': '', 'source_dataset_count': '', 'termination_status_json': '', 'action_norm_mean': '', 'q05': '', 'q50': '', 'q95': ''})
    wc(CAUSAL / 'subsets/SUPPORT_SUBSET_MANIFEST.csv', manifests)
    wc(CAUSAL / 'subsets/SUPPORT_SUBSET_COMPOSITION.csv', comps)
    wc(CAUSAL / 'subsets/SUPPORT_SEPARATION_METRICS.csv', seps)
    checks = []
    for r in (1, 2, 3):
        for c in ('0.2', '0.5', '0.8', '1.0'):
            m = [x for x in seps if x['repeat'] == r and x['eval_command_vx'] == c]
            d = {x['subset']: x['d5'] for x in m}
            checks.append({'repeat': r, 'command': c, 'matched_d5': d[f"MATCHED_{c.replace('.', 'p')}"], 'distant_d5': d[f"DISTANT_{c.replace('.', 'p')}"], 'separated': d[f"MATCHED_{c.replace('.', 'p')}"] < d[f"DISTANT_{c.replace('.', 'p')}"]})
    wc(CAUSAL / 'subsets/SUPPORT_SUBSET_OVERLAP.csv', [x for x in comps if str(x['subset']).startswith('OVERLAP')])
    (CAUSAL / 'subsets/SUPPORT_SUBSET_PROTOCOL.md').write_text('# Frozen subset construction\n\nMaterialized before adaptations. MATCHED/DISTANT use only 5-NN held-out support locations; UNIFORM/RANDOM use the frozen rules in the protocol. Support separation results are in `SUPPORT_SEPARATION_METRICS.csv`.\n')
    assert all((x['separated'] for x in checks)), checks
    print('SUBSETS_FROZEN', len(manifests), checks)
if __name__ == '__main__':
    main()
