"""Serial, resumable experiment driver. Every new artifact stays in this directory."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT = artifact_dir('positive_control_variance_v1')
REPO = PORTFOLIO_ROOT
RESULTS = ROOT
M0 = checkpoint_root() / 'models/go2_2026-02-12'
P = artifact_root() / 'module_combo_v1/mass8/policy/m0/checkpoint'
PY = __import__('os').environ.get('SIMDIST_PYTHON', __import__('sys').executable)
SEEDS = list(range(23501, 23601))

def run(label, script, arguments, done=None):
    if done and resolve_artifact_path(done).exists():
        print('REUSED', label, flush=True)
        return
    log = ROOT / 'logs' / (label + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [PY, '-B', str(REPO / script), *map(str, arguments)]
    ledger = ROOT / 'commands.jsonl'
    with ledger.open('a') as f:
        f.write(json.dumps({'label': label, 'command': command, 'started_unix': time.time(), 'log': str(log)}) + '\n')
    print('START', label, flush=True)
    with log.open('w') as f:
        subprocess.run(command, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=True, env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
    print('FINISH', label, flush=True)

def evaluate(label, out, checkpoint, terrain, friction, velocity, controller='mppi', seeds=SEEDS):
    completed = {json.loads(p.read_text())['seed'] for p in out.rglob('summary.json')} if out.exists() else set()
    if set(seeds).issubset(completed):
        print('REUSED', label, flush=True)
        return
    gap = 'mass8' if terrain == 'boxes' else 'source'
    run(label, 'experiments/positive_control_variance_v1/eval_batch.py', ['--gap', gap, '--terrain', terrain, '--friction', friction, '--forward-vel', velocity, '--controller', controller, '--checkpoint', checkpoint, '--seeds', *seeds, '--out-root', out])

def screening():
    candidates = []
    for friction in (0.15, 0.8, 0.1):
        for velocity in (0.5, 1.0):
            name = f'f{round(friction * 100):03d}_v{round(velocity * 10):02d}'
            root = RESULTS / 'screen' / name
            row = {'condition': name, 'friction': friction, 'velocity': velocity}
            for controller in ('mppi', 'expert'):
                evaluate('screen_' + name + '_' + controller, root / controller, M0, 'plane', friction, velocity, controller, SEEDS[:20])
                summaries = [json.loads(p.read_text()) for p in (root / controller).rglob('summary.json')]
                assert len(summaries) == 20 and sorted((s['seed'] for s in summaries)) == SEEDS[:20]
                row[controller + '_failures'] = sum((s['status'] == 'warmup_failed' or s['fall'] for s in summaries))
            row['eligible'] = row['expert_failures'] <= 1 and 8 <= row['mppi_failures'] <= 17
            row['distance'] = max(0, row['expert_failures'] - 1) + max(0, 8 - row['mppi_failures'], row['mppi_failures'] - 17)
            candidates.append(row)
            print('SCREEN', json.dumps(row), flush=True)
    eligible = [c for c in candidates if c['eligible']]
    pool = eligible or candidates
    chosen = min(pool, key=lambda c: (0 if eligible else c['distance'], 0 if (c['friction'], c['velocity']) == (0.15, 0.5) else 1, c['expert_failures'], abs(c['mppi_failures'] - 12.5), c['condition']))
    with (RESULTS / 'screening.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(candidates[0]))
        w.writeheader()
        w.writerows(candidates)
    selection = {**chosen, 'reason': 'requested eligibility range' if eligible else 'No eligible condition; minimum L1 failure-count distance to expert<=1 and m0 in [8,17], requested preference breaks ties.', 'candidates': candidates}
    (RESULTS / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print('SELECTED', json.dumps(chosen), flush=True)
    return chosen

def chain(label, root, initial, first_seed, train_seed, terrain, friction, velocity):
    previous = initial
    train_paths, heldout_paths = ([], [])
    for index, stage in enumerate(('10k', '30k')):
        start = first_seed + index * 100
        count = 10000 if index == 0 else 20000
        for offset in range(6):
            seed = start + offset
            steps = count // 6 + int(offset < count % 6)
            dest = root / 'collections' / stage / f'seed_{seed}'
            run(f'{label}_collect_{stage}_{seed}', 'experiments/positive_control_variance_v1/collect.py', ['--gap', 'mass8' if terrain == 'boxes' else 'source', '--terrain', terrain, '--friction', friction, '--forward-vel', velocity, '--checkpoint', previous, '--seed', seed, '--steps', steps, '--out', dest], dest / 'summary.json')
            (train_paths if offset < 5 else heldout_paths).append(dest / 'rollout.h5')
        training = root / 'training' / stage
        run(f'{label}_train_{stage}', 'experiments/extended_adaptation_v1/train_seed_split.py', ['--previous', previous, '--seed', train_seed + index, '--train-rollouts', *train_paths, '--heldout-rollouts', *heldout_paths, '--latest-heldout', heldout_paths[-1], '--out', training], training / 'summary.json')
        previous = training / 'checkpoint'
    return previous

def losses(label, models, dataset, out):
    arguments = []
    for name, path in models.items():
        arguments += ['--model', f'{name}={path}']
    arguments += ['--dataset', f'independent_m0={dataset}', '--out', out]
    run(label, 'experiments/module_combo_v1/evaluate_latent_datasets.py', arguments, out.with_name(out.stem + '_summary.csv'))

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    selected = screening()
    friction, velocity = (selected['friction'], selected['velocity'])
    flat = RESULTS / 'flat'
    models = {'m0': M0}
    for label, first, train in (('chain1', 43101, 43501), ('chain2', 44101, 44501)):
        models[label] = chain('flat_' + label, flat / label, M0, first, train, 'plane', friction, velocity)
    for offset in range(10):
        seed = 45301 + offset
        dest = flat / 'validation' / f'seed_{seed}'
        run(f'flat_validation_{seed}', 'experiments/positive_control_variance_v1/collect.py', ['--gap', 'source', '--terrain', 'plane', '--friction', friction, '--forward-vel', velocity, '--checkpoint', M0, '--seed', seed, '--steps', 500, '--out', dest], dest / 'summary.json')
    losses('flat_losses', models, flat / 'validation', flat / 'losses.csv')
    for name, checkpoint in models.items():
        if name == 'm0':
            import shutil
            source = RESULTS / 'screen' / selected['condition'] / 'mppi'
            destination = flat / 'evaluations/m0'
            destination.mkdir(parents=True, exist_ok=True)
            for ep in source.iterdir():
                if ep.is_dir() and (not (destination / ep.name).exists()):
                    shutil.copytree(ep, destination / ep.name)
        evaluate('flat_eval_' + name, flat / 'evaluations' / name, checkpoint, 'plane', friction, velocity)
    evaluate('flat_eval_base', flat / 'evaluations/base', M0, 'plane', friction, velocity, 'base')
    mass = RESULTS / 'mass8'
    models = {'m0': M0, 'old_chain1': artifact_root() / 'adaptation_curve_v1/training/stage_30000/checkpoint', 'old_chain2': artifact_root() / 'followup_v1/chain/training/30k/checkpoint', 'old_DP': artifact_root() / 'module_combo_v1/followup_repro/mass8_d_p/training/30k/checkpoint'}
    for label, initial, first, train in (('new_chain3', M0, 46101, 46501), ('new_chain4', M0, 47101, 47501), ('new_DP2', P, 48101, 48501)):
        checkpoint = chain('mass8_' + label, mass / label, initial, first, train, 'boxes', 0.8, 1.0)
        models[label] = checkpoint
        if label == 'new_DP2':
            dest = mass / label / 'original_policy'
            run('mass8_compose_DP2', 'experiments/module_combo_v1/compose_policy_checkpoint.py', ['--dynamics', checkpoint, '--policy', M0, '--out', dest], dest / 'summary.json')
            checkpoint = dest / 'checkpoint'
        evaluate('mass8_eval_' + label, mass / 'evaluations' / label, checkpoint, 'boxes', 0.8, 1.0)
    for offset in range(10):
        seed = 49301 + offset
        dest = mass / 'validation' / f'seed_{seed}'
        run(f'mass8_validation_{seed}', 'experiments/positive_control_variance_v1/collect.py', ['--gap', 'mass8', '--terrain', 'boxes', '--friction', 0.8, '--forward-vel', 1.0, '--checkpoint', M0, '--seed', seed, '--steps', 500, '--out', dest], dest / 'summary.json')
    losses('mass8_losses', models, mass / 'validation', mass / 'losses.csv')
    run('aggregate', 'analysis/positive_control_variance_v1/aggregate.py', [])
    (ROOT / 'COMPLETE.json').write_text(json.dumps({'completed_unix': time.time()}, indent=2) + '\n')
    print('ALL_COMPLETE', flush=True)
if __name__ == '__main__':
    main()
