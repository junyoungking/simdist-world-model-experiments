"""Build and resume the predeclared T3 r4/r5 replication without touching r1-r3."""
from __future__ import annotations
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import argparse, csv, hashlib, json, math, os, shutil, subprocess, sys
from pathlib import Path
import h5py, numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t
W = PORTFOLIO_ROOT
R = legacy_dir('11_t3_support_causal')
F = R / 'fresh_replication'
D = F / 'datasets'
P = legacy_dir('research_topic_validation_20260914T012121Z_continuation')
A = PORTFOLIO_ROOT
N = 500
CMDS = ('0p2', '0p5', '0p8', '1p0')
SEED = {4: 96104, 5: 96105}
COL = '10104,10105,10240,10241,10242,10250,10251,10252'

def wc(p, rows, fields=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def hs(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ''

def hdir(p):
    return {str(x.relative_to(p)): hs(x) for x in sorted(p.rglob('*')) if x.is_file()}

def source_paths(r):
    return [D / f't3f_narrow_r{r}', D / f't3f_broad_r{r}_v0p2', D / f't3f_broad_r{r}_v0p5', D / f't3f_broad_r{r}_v0p8']

def load_source(folder, model):
    man = json.loads((folder / 'dataset_manifest.json').read_text())
    p = resolve_artifact_path(man['processed_dir'])

    def rd(k):
        with h5py.File(p / f'{k}.hdf5', 'r') as f:
            return f['data'][:]
    prop, ext, act, cmd, starts = (rd('proprio_obs'), rd('extero_obs'), rd('acts'), rd('cmds'), rd('start_idxs').astype(int))
    import jax.numpy as jnp
    ctxs = []
    aa = []
    for lo in range(0, len(starts), 64):
        s = starts[lo:lo + 64]
        x = {'proprio_obs_hist': np.stack([prop[i:i + 26] for i in s]), 'extero_obs': ext[s + 25], 'acts_hist': np.stack([act[i:i + 25] for i in s]), 'fut_acts': np.stack([act[i + 25:i + 50] for i in s]), 'fut_cmds': np.stack([cmd[i + 25:i + 51] for i in s])}
        q = model.get_scaler().scale({k: jnp.asarray(v) for k, v in x.items()})
        e = model.encoder(q, deterministic=True)
        ctxs.append(np.asarray(np.concatenate([np.asarray(e['history']), np.asarray(e['latent'])[:, None, :]], axis=1), np.float32))
        aa.append(x['fut_acts'][:, 0].astype(np.float32))
    return (p, man, prop, ext, act, cmd, starts, np.concatenate(ctxs), np.concatenate(aa))

def materialize(name, ids, meta):
    out = D / name
    proc = out / 'processed_data_go2_H-25_T-25'
    out.mkdir()
    proc.mkdir()
    groups = []
    off = 0
    for folder in sorted({m['folder'] for m in meta}):
        p = resolve_artifact_path(json.loads((folder / 'dataset_manifest.json').read_text())['processed_dir'])
        n = len(meta[0]['prop']) if False else None
        with h5py.File(p / 'proprio_obs.hdf5', 'r') as f:
            n = len(f['data'])
        groups.append((folder, p, off))
        off += n
    for key in ('proprio_obs', 'extero_obs', 'acts', 'cmds', 'rewards', 'values', 'expert_policy_flag'):
        with h5py.File(proc / f'{key}.hdf5', 'w') as o:
            for _, p, _ in groups:
                with h5py.File(p / f'{key}.hdf5', 'r') as f:
                    a = f['data'][:]
                if 'data' not in o:
                    ds = o.create_dataset('data', shape=(0,) + a.shape[1:], maxshape=(None,) + a.shape[1:], dtype=a.dtype)
                ds.resize(len(ds) + len(a), axis=0)
                ds[-len(a):] = a
    offsets = {str(x): z for x, _, z in groups}
    start = []
    ident = []
    for i in ids:
        m = meta[int(i)]
        start.append(int(m['start']) + offsets[str(m['folder'])])
        ident.append({'source_dataset': m['folder'].name, 'start_idx': int(m['start']), 'source_seed': m['seed']})
    with h5py.File(proc / 'start_idxs.hdf5', 'w') as f:
        f.create_dataset('data', data=np.asarray(start, dtype=np.int64))
    for k in ('scaler_params.json', 'system.yaml', 'dataset_metrics.json'):
        shutil.copy2(groups[0][1] / k, proc / k)
    man = {'name': name, 'role': 'collection', 'view_kind': 't3_fresh_equal_size_subset', 'eligible_windows': N, 'seeds': sorted({m['seed'] for m in meta}), 'processed_dir': str(proc), 'processed_hashes': hdir(proc), 'window_identities': ident}
    (out / 'dataset_manifest.json').write_text(json.dumps(man, indent=2) + '\n')
    return man

def fps(x, meta):
    first = min(range(len(x)), key=lambda i: (meta[i]['folder'].name, int(meta[i]['start'])))
    chosen = [first]
    best = np.sum((x - x[first]) ** 2, axis=1)
    for _ in range(1, N):
        j = int(np.argmax(best))
        chosen.append(j)
        best = np.minimum(best, np.sum((x - x[j]) ** 2, axis=1))
    return np.asarray(chosen, dtype=int)

def build():
    from simdist.utils.model import load_model_from_ckpt
    model, _, step = load_model_from_ckpt(str(checkpoint_root() / 'models/go2_2026-02-12'))
    assert step == 367004
    z = np.load(R / '../11_t3_target_support/existing_t3_audit/EXISTING_T3_SUPPORT_VECTORS.npz')
    mean, std, radius = (z['normalization_mean'], z['normalization_std'], float(z['density_radius']))
    rows = []
    seps = []
    summary = []
    for r in (4, 5):
        meta = []
        xx = []
        for folder in source_paths(r):
            p, man, prop, ext, act, cmd, starts, ctx, a = load_source(folder, model)
            xx.append(np.concatenate([ctx.reshape(len(ctx), -1), a], axis=1))
            meta += [{'folder': folder, 'start': s, 'seed': man['seed']} for s in starts]
        x = np.concatenate(xx)
        xn = (x - mean) / std
        rng = np.random.default_rng(95100 + r)
        sets = {'UNIFORM': fps(xn, meta), 'RANDOM': rng.choice(len(xn), N, replace=False)}
        for c in CMDS:
            ev = []
            for er in (1, 2, 3):
                ev.append(np.concatenate([z[f'eval_cov_eval_v{c}_r{er}_context'].reshape(len(z[f'eval_cov_eval_v{c}_r{er}_context']), -1), z[f'eval_cov_eval_v{c}_r{er}_action']], axis=1))
            ev = (np.concatenate(ev) - mean) / std
            d = []
            for lo in range(0, len(xn), 64):
                d.extend(np.partition(cdist(xn[lo:lo + 64], ev), 4, axis=1)[:, :5].mean(1))
            o = np.argsort(d, kind='stable')
            sets[f'MATCHED_{c}'] = o[:N]
            sets[f'DISTANT_{c}'] = o[-N:]
        for label, ids in sets.items():
            name = f't3f_r{r}_{label.lower()}'
            man = materialize(name, ids, meta) if not (D / name).exists() else json.loads((D / name / 'dataset_manifest.json').read_text())
            rows.append({'repeat': r, 'subset': label, 'dataset': name, 'selected_windows': N, 'manifest_sha256': hs(D / name / 'dataset_manifest.json')})
        for c in CMDS:
            ev = []
            for er in (1, 2, 3):
                ev.append(np.concatenate([z[f'eval_cov_eval_v{c}_r{er}_context'].reshape(len(z[f'eval_cov_eval_v{c}_r{er}_context']), -1), z[f'eval_cov_eval_v{c}_r{er}_action']], axis=1))
            ev = (np.concatenate(ev) - mean) / std
            for s in (f'MATCHED_{c}', f'DISTANT_{c}'):
                q = cdist(ev, xn[sets[s]])
                p = np.partition(q, 9, axis=1)[:, :10]
                p.sort(1)
                seps += [{'repeat': r, 'eval_command_vx': c.replace('p', '.'), 'subset': s, 'knn_d1': float(p[:, 0].mean()), 'knn_d5': float(p[:, 4].mean()), 'knn_d10': float(p[:, 9].mean()), 'fixed_radius_neighbors': float((q <= radius).sum(1).mean())}]
        summary.append({'repeat': r, 'pool_windows': len(xn), 'minimum_required': N, 'sufficient': len(xn) >= N, 'source_datasets_json': json.dumps([x.name for x in source_paths(r)])})
    wc(F / 'FRESH_SUBSET_MANIFEST.csv', rows)
    wc(F / 'FRESH_SUPPORT_SEPARATION.csv', seps)
    wc(F / 'FRESH_POOL_SUPPORT_SUMMARY.csv', summary)

def run(cmd):
    env = os.environ | {'SIMDIST_C_OUTPUT_ROOT': str(F), 'SIMDIST_C_EXPERIMENT_MODE': 'mass', 'PYTHONPATH': str(A)}
    return subprocess.run([sys.executable, '-u', *map(str, cmd)], cwd=A, env=env).returncode

def execute():
    rows = list(csv.DictReader((F / 'FRESH_SUBSET_MANIFEST.csv').open()))
    out = []
    raw = []
    evals = [f'cov_eval_v{c}_r{e}' for c in CMDS for e in (1, 2, 3)]
    for r in rows:
        rn = f"t3f_{r['dataset']}_K1000"
        ad = F / 'adaptation' / rn
        mo = F / 'models' / rn
        ev = F / 'evaluation' / rn
        if not (ad / 'result.json').is_file():
            assert run([PORTFOLIO_ROOT / 'experiments/simdist_c/adapt.py', '--name', r['dataset'], '--run', rn, '--seed', SEED[int(r['repeat'])], '--steps', '1000', '--allowed-training-seeds', '96104,96105', '--allowed-collection-seeds', COL, '--training-dataset-root', D, '--heldout-name', 'cov_eval_v0p5_r1', '--heldout-dataset-root', P / 'datasets', '--log-dir', F / 'adaptation_curves' / rn]) == 0
        if not (ev / 'metrics.json').is_file():
            assert run([PORTFOLIO_ROOT / 'experiments/simdist_c/evaluate.py', '--names', *evals, '--checkpoint', mo, '--output', ev, '--dataset-root', P / 'datasets']) == 0
        result = json.loads((ad / 'result.json').read_text())
        m = json.loads((ev / 'metrics.json').read_text())
        assert result['actual_optimizer_updates'] == 1000 and (not result['frozen_changes']) and (len(m['per_seed']) == 12)
        out.append({'repeat': r['repeat'], 'subset': r['subset'], 'dataset': r['dataset'], 'run': rn, 'checkpoint': result['checkpoint'], 'status': 'VALID_COMPLETE', 'adaptation_sha256': hs(ad / 'result.json'), 'evaluation_sha256': hs(ev / 'metrics.json')})
        raw += [{'repeat': r['repeat'], 'subset': r['subset'], 'eval_command_vx': x['name'].split('_')[2][1:].replace('p', '.'), 'latent_mse': x['latent_mse']} for x in m['per_seed']]
    wc(F / 'FRESH_ADAPTATION_RUN_MANIFEST.csv', out)
    wc(F / 'FRESH_REPLICATION_RESULTS.csv', raw)

def finalize():
    h = list(csv.DictReader((R / 'adaptation/SUPPORT_ADAPTATION_RAW.csv').open()))
    f = list(csv.DictReader((F / 'FRESH_REPLICATION_RESULTS.csv').open()))
    rows = []
    for c in ('0.2', '0.5', '0.8', '1.0'):
        vals = []
        for r in range(1, 6):
            q = h if r < 4 else f
            tag = c.replace('.', 'p')
            a = [float(x['latent_mse']) for x in q if int(x['repeat']) == r and x['subset'] == f'MATCHED_{tag}' and (x['eval_command_vx'] == c)]
            b = [float(x['latent_mse']) for x in q if int(x['repeat']) == r and x['subset'] == f'DISTANT_{tag}' and (x['eval_command_vx'] == c)]
            vals.append(float(np.mean(b) - np.mean(a)))
        a = np.asarray(vals)
        sd = float(a.std(ddof=1))
        se = sd / math.sqrt(5)
        m = float(t.ppf(0.975, 4) * se)
        rows.append({'command_vx': c, 'raw_n5_delta_md_json': json.dumps(vals), 'historical_n3_mean': float(a[:3].mean()), 'fresh_r4_r5_json': json.dumps(vals[3:]), 'mean': float(a.mean()), 'median': float(np.median(a)), 'sd': sd, 'se': se, 'ci95_low': float(a.mean() - m), 'ci95_high': float(a.mean() + m), 'positive_sign_count': int((a > 0).sum()), 'min': float(a.min()), 'max': float(a.max())})
    wc(F / 'FRESH_REPLICATION_SUMMARY.csv', rows)
    (F / 'FRESH_REPLICATION_SUMMARY.md').write_text('# T3 fresh replication summary\n\n' + '\n'.join((f"- vx={x['command_vx']}: n=5 mean Δ_MD={x['mean']:.8g}; fresh r4/r5={x['fresh_r4_r5_json']}" for x in rows)) + '\n')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('build', 'execute', 'finalize'))
    a = p.parse_args()
    globals()[a.phase]()
