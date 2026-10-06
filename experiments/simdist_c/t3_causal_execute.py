"""Safely resume the frozen 30-cell T3 f-only causal matrix."""
from __future__ import annotations
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import csv, datetime, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
WORK = PORTFOLIO_ROOT
REPO = PORTFOLIO_ROOT
ROOT = legacy_dir('11_t3_support_causal')
PRIOR = legacy_dir('research_topic_validation_20260914T012121Z_continuation')
PY = sys.executable
allowed = '10101,10102,10103,10210,10211,10212,10220,10221,10222,10230,10231,10232'
tseeds = {1: 96101, 2: 96102, 3: 96103}
EVAL_NAMES = [f'cov_eval_v{c}_r{e}' for c in ('0p2', '0p5', '0p8', '1p0') for e in (1, 2, 3)]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ''

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')

def wc(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def run(label, args):
    logs = ROOT / 'runtime_logs'
    logs.mkdir(exist_ok=True)
    out = logs / (label + '.stdout.log')
    err = logs / (label + '.stderr.log')
    start = time.time()
    with out.open('w') as so, err.open('w') as se:
        p = subprocess.run([PY, '-u', *map(str, args)], cwd=REPO, stdout=so, stderr=se, env=os.environ | {'SIMDIST_C_OUTPUT_ROOT': str(ROOT), 'SIMDIST_C_EXPERIMENT_MODE': 'mass', 'PYTHONPATH': str(REPO)})
    if p.returncode:
        raise RuntimeError(f'{label} failed (exit={p.returncode}); see {err}')
    return time.time() - start

def valid_adapt(runname):
    p = ROOT / 'adaptation' / runname / 'result.json'
    model = ROOT / 'models' / runname
    try:
        x = json.loads(p.read_text())
        ok = x['initial_step'] == 367004 and x['final_step'] == 368004 and (x['actual_optimizer_updates'] == 1000) and (not x['frozen_changes']) and model.is_dir()
        return (bool(ok), x)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return (False, {})

def valid_eval(runname):
    p = ROOT / 'evaluation' / runname / 'metrics.json'
    try:
        x = json.loads(p.read_text())
        got = [z['name'] for z in x['per_seed']]
        return (bool(x.get('external') and got == EVAL_NAMES and (len(got) == 12)), x)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return (False, {})

def qlog(original, reason):
    if not original.exists():
        return None
    dest = original.with_name(original.name + '.__incomplete_' + stamp())
    files = [str(p.relative_to(original)) for p in sorted(original.rglob('*')) if p.is_file()]
    shutil.move(str(original), str(dest))
    log = ROOT / 'repair/T3_QUARANTINE_LOG.md'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        f.write(f'\n## {stamp()}\n\n- Original: `{original}`\n- Quarantine: `{dest}`\n- Files found: `{files}`\n- Files missing: `metrics.json` or a valid completion manifest\n- Reason: {reason}\n')
    return dest

def audit(rows):
    out = []
    for row in rows:
        r = int(row['repeat'])
        subset = row['subset']
        ds = row['dataset']
        runname = f't3c_{ds}_K1000'
        ad = ROOT / 'adaptation' / runname
        ev = ROOT / 'evaluation' / runname
        model = ROOT / 'models' / runname
        va, ar = valid_adapt(runname)
        ve, em = valid_eval(runname)
        if va and ve:
            status = 'VALID_COMPLETE'
        elif va:
            status = 'VALID_ADAPT_ONLY' if not ev.exists() else 'INCOMPLETE_EVAL'
        elif ad.exists() or model.exists():
            status = 'INCOMPLETE_ADAPT'
        else:
            status = 'MISSING'
        paths = [p for p in (ad / 'result.json', model, ev / 'metrics.json') if p.exists()]
        ts = max((p.stat().st_mtime for p in paths), default=0)
        out.append({'repeat': r, 'subset_condition': subset, 'dataset': ds, 'command_identity': f'adapt.py --name {ds} --run {runname} --seed {tseeds[r]} --steps 1000; evaluate.py --names ' + ' '.join(EVAL_NAMES), 'adaptation_directory': str(ad), 'adaptation_manifest_exists': (ad / 'result.json').exists(), 'checkpoint_directory_exists': model.is_dir(), 'adaptation_complete': va, 'evaluation_directory': str(ev), 'metrics_json_exists': (ev / 'metrics.json').exists(), 'evaluation_manifest_exists': (ev / 'metrics.json').exists(), 'evaluation_complete': ve, 'timestamp_utc': datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat() if ts else '', 'adaptation_result_sha256': sha(ad / 'result.json'), 'evaluation_metrics_sha256': sha(ev / 'metrics.json'), 'status': status})
    wc(ROOT / 'repair/T3_PARTIAL_ARTIFACT_AUDIT.csv', out)
    counts = {s: sum((x['status'] == s for x in out)) for s in sorted({x['status'] for x in out})}
    lines = ['# T3 partial artifact audit', '', f'Frozen expected cells: **{len(out)}**.', '', '## Status counts', ''] + [f'- {k}: {v}' for k, v in counts.items()] + ['', 'Directory presence alone was not treated as completion. `metrics.json` is the evaluation completion manifest. Checksums are SHA-256 of manifest files.']
    (ROOT / 'repair/T3_PARTIAL_ARTIFACT_AUDIT.md').write_text('\n'.join(lines) + '\n')
    return out

def main():
    rows = list(csv.DictReader((ROOT / 'subsets/SUPPORT_SUBSET_MANIFEST.csv').open()))
    audit(rows)
    (ROOT / 'repair/T3_QUARANTINE_LOG.md').parent.mkdir(parents=True, exist_ok=True)
    (ROOT / 'repair/T3_QUARANTINE_LOG.md').touch(exist_ok=True)
    records = []
    manifest = []
    for row in rows:
        r = int(row['repeat'])
        subset = row['subset']
        ds = row['dataset']
        runname = f't3c_{ds}_K1000'
        ad = ROOT / 'adaptation' / runname
        model = ROOT / 'models' / runname
        ev = ROOT / 'evaluation' / runname
        va, result = valid_adapt(runname)
        wall = 0.0
        if not va:
            qlog(ad, 'Adaptation manifest/checkpoint failed frozen-completion validation.')
            qlog(model, 'Associated incomplete/corrupt adaptation checkpoint.')
            qlog(ROOT / 'adaptation_curves' / runname, 'Associated incomplete/corrupt adaptation curve artifacts.')
            wall = run('adapt_' + runname, [PORTFOLIO_ROOT / 'experiments/simdist_c/adapt.py', '--name', ds, '--run', runname, '--seed', tseeds[r], '--steps', '1000', '--allowed-training-seeds', ','.join(map(str, tseeds.values())), '--allowed-collection-seeds', allowed, '--training-dataset-root', ROOT / 'datasets', '--heldout-name', f'cov_eval_v0p5_r{r}', '--heldout-dataset-root', PRIOR / 'datasets', '--logging-eval-interval', '100', '--log-dir', ROOT / 'adaptation_curves' / runname])
            va, result = valid_adapt(runname)
            if not va:
                raise RuntimeError(f'Adaptation did not produce a valid frozen completion: {runname}')
        ve, metrics = valid_eval(runname)
        if not ve and ev.exists():
            qlog(ev, 'Evaluation directory exists without valid metrics.json / frozen 12-cell completion.')
        if not ve:
            run('eval_' + runname, [PORTFOLIO_ROOT / 'experiments/simdist_c/evaluate.py', '--names', *EVAL_NAMES, '--checkpoint', model, '--output', ev, '--dataset-root', PRIOR / 'datasets'])
            ve, metrics = valid_eval(runname)
            if not ve:
                raise RuntimeError(f'Evaluation did not produce a valid frozen completion: {runname}')
        manifest.append({'repeat': r, 'subset': subset, 'dataset': ds, 'run': runname, 'adaptation_result_sha256': sha(ad / 'result.json'), 'checkpoint': result['checkpoint'], 'actual_optimizer_updates': result['actual_optimizer_updates'], 'evaluation_metrics_sha256': sha(ev / 'metrics.json'), 'status': 'VALID_COMPLETE'})
        for m in metrics['per_seed']:
            records.append({'repeat': r, 'subset': subset, 'dataset': ds, 'run': runname, 'eval_dataset': m['name'], 'eval_command_vx': m['name'].split('_')[2][1:].replace('p', '.'), 'eval_seed': m['seed'], 'latent_mse': m['latent_mse'], 'optimizer_updates': result['actual_optimizer_updates'], 'wall_seconds_adapt': wall, 'checkpoint': result['checkpoint']})
    wc(ROOT / 'adaptation/SUPPORT_ADAPTATION_RUN_MANIFEST.csv', manifest)
    wc(ROOT / 'adaptation/SUPPORT_ADAPTATION_RAW.csv', records)
    print('CAUSAL_EXECUTION_COMPLETE', len(manifest), 'adaptations', len(records), 'evaluation rows', flush=True)
if __name__ == '__main__':
    main()
