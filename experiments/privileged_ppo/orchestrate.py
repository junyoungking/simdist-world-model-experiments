"""Sequential train/evaluation orchestration; no automatic failed-seed replacement."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import os
import subprocess
import sys
import time
ROOT = artifact_dir('privileged_ppo')
REPO = PORTFOLIO_ROOT
PYTHON = sys.executable
inventory = json.loads(document_path('privileged_ppo', 'CHECKPOINT_HASHES.json').read_text())
ledger_path = ROOT / '09_provenance/EXECUTION_LEDGER.json'
ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []

def run(label, argv, artifact, env=None):
    if artifact.exists():
        data = json.loads(artifact.read_text())
        assert data.get('status') in ['PASS', 'OPERATIONAL_PLATEAU_CONFIRMED'] or data.get('finite') is True
        return
    log_path = ROOT / '09_provenance' / f'{label}.log'
    attempt = 1
    while log_path.exists():
        attempt += 1
        log_path = ROOT / '09_provenance' / f'{label}_attempt{attempt}.log'
    entry = {'label': label, 'argv': argv, 'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING', 'log_path': str(log_path), 'attempt': attempt}
    ledger.append(entry)
    (ROOT / '09_provenance/EXECUTION_LEDGER.json').write_text(json.dumps(ledger, indent=2) + '\n')
    with log_path.open('x') as f:
        proc = subprocess.run(argv, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', **(env or {})})
    entry['ended_utc'] = datetime.now(timezone.utc).isoformat()
    entry['exit_code'] = proc.returncode
    entry['status'] = 'PASS' if artifact.exists() else 'FAILED_MISSING_COMPLETION'
    (ROOT / '09_provenance/EXECUTION_LEDGER.json').write_text(json.dumps(ledger, indent=2) + '\n')
    assert artifact.exists(), f'Failed execution {label}; inspect recorded failure without changing seed/config'
    data = json.loads(artifact.read_text())
    assert data.get('status') in ['PASS', 'OPERATIONAL_PLATEAU_CONFIRMED'] or data.get('finite') is True

def mppi(category, model, seed, mass, steps=1000, role='control'):
    data_root = ROOT / category / 'mppi'
    identity = f'{model}_{seed}'
    log_dir = ROOT / category / f'raw_{model}_{seed}'
    argv = [PYTHON, '-u', str(PORTFOLIO_ROOT / 'experiments/privileged_ppo/evaluate_mppi.py'), '--name', identity, '--family', 'added_mass', '--friction', '.8', '--added-mass', str(mass), '--kp', '25', '--kd', '.5', '--seed', str(seed), '--steps', str(steps), '--role', role, '--checkpoint', inventory[model]['path'], '--log-dir', str(log_dir)]
    run(f'{category}_{identity}', argv, data_root / 'datasets' / identity / 'rollout.json', {'SIMDIST_C_EXPERIMENT_MODE': 'mass', 'SIMDIST_C_OUTPUT_ROOT': str(data_root)})

def ppo(category, model, checkpoint, seeds, mass=8.0, stochastic=False):
    out = ROOT / category / model
    argv = [PYTHON, '-u', str(PORTFOLIO_ROOT / 'experiments/privileged_ppo/evaluate_ppo.py'), '--checkpoint', str(checkpoint), '--seeds', ','.join(map(str, seeds)), '--output', str(out), '--mass', str(mass)]
    if stochastic:
        argv.append('--stochastic')
    run(f'{category}_{model}', argv, out / 'result.json')
try:
    while not (ROOT / '02_training/seed_62001/result.json').exists():
        status_path = ROOT / '02_training/seed_62001/STATUS.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        assert status.get('phase') != 'FAILED', 'First training seed failed; preserve failure and resume same seed/config'
        time.sleep(5)
    for seed in [62002, 62003]:
        argv = [PYTHON, '-u', str(PORTFOLIO_ROOT / 'experiments/privileged_ppo/train_target.py'), '--seed', str(seed)]
        folder = ROOT / '02_training' / f'seed_{seed}'
        if not (folder / 'result.json').exists() and folder.exists():
            checkpoints = sorted((folder / 'checkpoints').glob('target_*.pt'))
            assert checkpoints, 'No valid resume checkpoint; preserve failed state'
            argv += ['--resume', str(checkpoints[-1])]
        run(f'train_{seed}', argv, ROOT / '02_training' / f'seed_{seed}' / 'result.json')
    for model in ['MB1', 'MB2', 'MB3']:
        mppi('01_smoke', model, 64001, 8.0, steps=50, role='smoke')
    for seed in range(66001, 66011):
        mppi('04_0kg_anchor', 'M0', seed, 0.0)
    ppo('04_0kg_anchor', 'source_PPO', checkpoint_root() / 'rl/go2_2026-02-11/model_4999.pt', range(66001, 66011), mass=0.0)
    for model in ['M0', 'MB1', 'MB2', 'MB3']:
        for seed in range(61001, 61011):
            mppi('05_mass8_primary', model, seed, 8.0)
    for seed in [62001, 62002, 62003]:
        training = json.loads((ROOT / '02_training' / f'seed_{seed}' / 'result.json').read_text())
        final = resolve_artifact_path(training['final']['checkpoint'])
        ppo('05_mass8_primary', f'PPO_{seed}', final, range(61001, 61011))
        ppo('06_critic', f'PPO_{seed}', final, range(67001, 67011), stochastic=True)
    subprocess.run([PYTHON, str(PORTFOLIO_ROOT / 'analysis/privileged_ppo/summarize.py')], cwd=REPO, check=True)
except BaseException:
    import traceback
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (ROOT / '09_provenance' / f'ORCHESTRATION_FAILURE_{stamp}.txt').write_text(traceback.format_exc())
    raise
