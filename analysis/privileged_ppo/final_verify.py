"""Read-only final evidence checks, with a new-root verification artifact."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np
ROOT = artifact_dir('privileged_ppo')

def read(p):
    return json.loads(resolve_artifact_path(p).read_text())

def rows(p):
    return list(csv.DictReader(resolve_artifact_path(p).open()))
checks = {}
protocol = read(document_path('privileged_ppo', 'PROTOCOL_HASHES.json'))
checks['scientific_protocol_unchanged'] = hashlib.sha256(document_path('privileged_ppo', 'PROTOCOL.md').read_bytes()).hexdigest() == protocol['protocol_sha256']
inventory = read(document_path('privileged_ppo', 'CHECKPOINT_HASHES.json'))
for model in ['M0', 'MB1', 'MB2', 'MB3']:
    cp = inventory[model]
    checks[f'{model}_all_checkpoint_files_unchanged'] = all((hashlib.sha256((resolve_artifact_path(cp['path']) / name).read_bytes()).hexdigest() == expected for name, expected in cp['file_sha256'].items()))
source = read(ROOT / '09_provenance/SOURCE_FILE_HASHES.json')
checks['all402_source_config_raw_assets_unchanged'] = len(source) == 402 and all((resolve_artifact_path(name).is_file() and hashlib.sha256(resolve_artifact_path(name).read_bytes()).hexdigest() == data['sha256'] for name, data in source.items()))
primary = rows(ROOT / '05_mass8_primary/MASS8_PRIMARY_EVAL_RAW.csv')
anchor = rows(ROOT / '04_0kg_anchor/0KG_SOURCE_ANCHOR.csv')
checks['primary70_unique_complete_cells'] = len(primary) == 70 and len({(r['model'], int(r['eval_seed'])) for r in primary}) == 70
checks['anchor20_unique_complete_cells'] = len(anchor) == 20 and len({(r['model'], int(r['eval_seed'])) for r in anchor}) == 20
checks['complete1000_finite_return_tracking'] = all((int(r['executed_steps']) == 1000 and np.isfinite(float(r['Return'])) and np.isfinite(float(r['vx_RMSE'])) for r in primary + anchor))
checks['all_new_roles_disjoint'] = all((r['status'] == 'PASS' and int(r['overlap_count']) == 0 for r in rows(ROOT / '07_summary/HELDOUT_LEAKAGE_AUDIT.csv')))
checks['initial_state_terrain_and_semantic_pairing'] = all((all((r[k] == 'True' for k in ['initial_state_sha256', 'terrain_sha256', 'reward_signature', 'termination_signature', 'action_signature'])) for r in rows(ROOT / '07_summary/PAIRED_RUNTIME_CONTRACT_AUDIT.csv')))
training = []
for seed in [62001, 62002, 62003]:
    result = read(ROOT / '02_training' / f'seed_{seed}' / 'result.json')
    training.append(result)
    checks[f'PPO{seed}_valid_final_and_budget_counters'] = result['status'] == 'OPERATIONAL_PLATEAU_CONFIRMED' and result['target_env_steps'] == result['target_iterations'] * 4096 * 24 and (result['target_optimizer_updates'] == result['target_iterations'] * 20) and (hashlib.sha256(resolve_artifact_path(result['final']['checkpoint']).read_bytes()).hexdigest() == result['final']['checkpoint_sha256'])
    agent = read(ROOT / '02_training' / f'seed_{seed}' / 'effective_agent.json')
    agent.pop('seed')
    if seed == 62001:
        reference_agent = agent
    checks[f'PPO{seed}_same_hyperparameters'] = agent == reference_agent
    curves = rows(ROOT / '02_training' / f'seed_{seed}' / 'PPO_TRAINING_CURVES.csv')
    checks[f'PPO{seed}_validation_frozen_roles_only'] = all((int(r['eval_seed']) in range(65001, 65011) and int(r['train_seed']) == seed for r in curves))
    pairs = {(int(r['target_iteration']), int(r['eval_seed'])) for r in curves}
    checks[f'PPO{seed}_validation_no_missing_or_duplicate'] = len(pairs) == len(curves) and pairs == {(i, e) for i in range(0, result['target_iterations'] + 1, 250) for e in range(65001, 65011)}
readback_count = 0
semantic_hashes = {k: set() for k in ['reward_signature', 'termination_signature', 'action_signature', 'command_signature']}
for category, mass in [('03_validation', 8.0), ('04_0kg_anchor', 0.0), ('05_mass8_primary', 8.0), ('06_critic', 8.0)]:
    paths = list((ROOT / category).glob('**/RUNTIME_READBACK.json')) + list((ROOT / category).glob('**/full_runtime_readback.json'))
    for p in paths:
        for rb in read(p):
            readback_count += 1
            for k in semantic_hashes:
                semantic_hashes[k].add(rb[k])
            assert rb['mass_inertia_PASS'] and np.isclose(rb['base_mass_min'], 6.921000480651855 + mass, atol=1e-06, rtol=0)
            assert rb['kp'] == [25.0, 25.0] and rb['kd'] == [0.5, 0.5]
            assert rb['physics_dt'] == 0.005 and rb['control_dt'] == 0.02 and (rb['reference_vx'] == 0.5)
            assert np.allclose(rb['foot_material_min'], [0.8, 0.8, 0.05], atol=1e-06, rtol=0)
            assert np.allclose(rb['foot_material_max'], [0.8, 0.8, 0.05], atol=1e-06, rtol=0)
checks['every_recorded_initial_and_reset_physics_readback'] = readback_count > 120
checks['every_command_reward_termination_action_signature_shared'] = all((len(values) == 1 for values in semantic_hashes.values()))
for category, records in [('04_0kg_anchor', anchor), ('05_mass8_primary', primary)]:
    for seed in sorted({int(r['eval_seed']) for r in records}):
        group = [r for r in records if int(r['eval_seed']) == seed]
        hashes = []
        for r in group:
            if r['model'] in ['M0', 'MB1', 'MB2', 'MB3']:
                rb = read(ROOT / category / 'mppi/datasets' / f"{r['model']}_{seed}" / 'full_runtime_readback.json')[0]
            else:
                rb = next((x for x in read(ROOT / category / r['model'] / 'RUNTIME_READBACK.json') if x['reason'] == f'seed_{seed}_initial'))
            hashes.append(rb['joint_friction_sha256'])
        checks[f'{category}_seed{seed}_initial_joint_friction_paired'] = len(set(hashes)) == 1
checks['three_training_seeds_retained'] = len(training) == 3
ledger = read(ROOT / '09_provenance/EXECUTION_LEDGER.json')
first_primary = min((x['started_utc'] for x in ledger if x['label'].startswith('05_mass8_primary')))
checks['all_final_selection_precedes_primary_test'] = all((read(ROOT / '05_mass8_primary' / f'PPO_{seed}' / 'result.json')['completed_utc'] > first_primary and resolve_artifact_path(ROOT / '02_training' / f'seed_{seed}' / 'result.json').stat().st_mtime < datetime.fromisoformat(first_primary).timestamp() for seed in [62001, 62002, 62003]))
verification = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checked_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks, 'readback_records': readback_count, 'historical_source_training_seed_overlap': 'UNVERIFIABLE_ORIGINAL_SOURCE_SEED_LOGS_ABSENT', 'resume_replay': 'seed62001_environment_reconstructed; not_exact_replay'}
(ROOT / '09_provenance/FINAL_VERIFICATION.json').write_text(json.dumps(verification, indent=2) + '\n')
assert verification['status'] == 'PASS', verification
print('FINAL_VERIFICATION_PASS', len(checks), 'checks', readback_count, 'readbacks')
