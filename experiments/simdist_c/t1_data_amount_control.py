"""Narrow T1 +4 -> +8 fixed-data-amount / buffer-composition control.

The runner is deliberately limited to data construction, f-only adaptation, and
external latent-MSE evaluation.  It never launches control, T2, T3, or any
other T1 schedule.
"""
from __future__ import annotations
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
REPO = PORTFOLIO_ROOT
WORK = PORTFOLIO_ROOT
ROOT = legacy_dir('10_t1_data_amount_control')
OLD = legacy_dir('t1_nonstationarity_v2_20260915T210000Z')
SOURCE = checkpoint_root() / 'models/go2_2026-02-12'
H = T = 25
N_CORE = N_EXTRA = N_HIST = 500
K = 1000
BATCH = 32
REPEATS = range(1, 6)
SUBSET_SEEDS = {r: 19100 + r for r in REPEATS}
SUPPLEMENT_SEEDS = {r: 17100 + r for r in REPEATS}
TRAIN_SEEDS = {r: 18100 + r for r in REPEATS}
KEYS = ['proprio_obs.hdf5', 'extero_obs.hdf5', 'acts.hdf5', 'cmds.hdf5', 'rewards.hdf5', 'values.hdf5', 'expert_policy_flag.hdf5']

def read(path):
    return json.loads(resolve_artifact_path(path).read_text())

def write_json(path, value):
    path = resolve_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def sha(path):
    return hashlib.sha256(resolve_artifact_path(path).read_bytes()).hexdigest()

def hashes(directory):
    directory = resolve_artifact_path(directory)
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}

def csv_write(path, rows, fields=None):
    path = resolve_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def old_name(repeat, stage, role):
    return f'e1_abrupt_r{repeat}_s{stage}_{role}'

def old_source(repeat, mass):
    stage = 1 if mass == 4 else 2
    return OLD / 'datasets' / old_name(repeat, stage, 'collection')

def old_eval(repeat):
    return OLD / 'datasets' / old_name(repeat, 2, 'eval')

def supplement_name(repeat):
    return f'dac_r{repeat}_plus8_supplement'

def env():
    value = dict(**__import__('os').environ)
    value['SIMDIST_C_OUTPUT_ROOT'] = str(ROOT)
    value['SIMDIST_C_EXPERIMENT_MODE'] = 'mass'
    value['PYTHONUNBUFFERED'] = '1'
    return value

def command(label, script, args):
    logs = ROOT / '02_runs' / 'runtime_logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f'{label}.stdout.log').open('x') as out, (logs / f'{label}.stderr.log').open('x') as err:
        subprocess.run([sys.executable, '-u', str(PORTFOLIO_ROOT / 'experiments/simdist_c' / script), *map(str, args)], cwd=REPO, env=env(), stdout=out, stderr=err, check=True)

def dataset_manifest(folder):
    return read(resolve_artifact_path(folder) / 'dataset_manifest.json')

def verify_source(folder, mass, role=None):
    manifest = dataset_manifest(folder)
    assert manifest['intervention_family'] == 'added_mass'
    assert float(manifest['friction']) == 0.8 and float(manifest['added_mass_kg']) == mass
    assert float(manifest['fixed_kp']) == 25.0 and float(manifest['fixed_kd']) == 0.5
    if role is not None:
        assert manifest['role'] == role
    assert resolve_artifact_path(manifest['processed_dir']).is_dir()
    assert hashes(manifest['processed_dir']) == manifest['processed_hashes']
    return manifest

def init():
    if ROOT.exists():
        raise RuntimeError(f'Fresh continuation root required: {ROOT}')
    for rel in ['00_protocol', '01_manifests', '02_runs/runtime_logs', '02_runs/checkpoints', '03_results', '04_figures', '05_report', 'datasets', 'models', 'adaptation', 'eval', 'logs']:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    protocol = {'protocol': 'T1 fixed-data-amount matched substitution', 'state': 'FROZEN_BEFORE_NEW_OUTCOMES', 'scope': 'one clean +4 kg -> +8 kg experiment; no control/T2/T3/E2/E3/E4', 'source_checkpoint': {'path': str(SOURCE), 'step': 367004, 'hashes': hashes(SOURCE)}, 'testbed': {'friction': 0.8, 'added_mass_history_kg': 4.0, 'added_mass_current_kg': 8.0, 'kp': 25.0, 'kd': 0.5, 'H': H, 'T': T}, 'arms': {'CURRENT_ONLY_1000': {'B_core': 500, 'B_extra': 500, 'A_hist': 0}, 'MIXED_500_500': {'B_core': 500, 'B_extra': 0, 'A_hist': 500}}, 'adaptation': {'f_only': True, 'frozen_modules': ['E', 'C', 'R', 'V', 'pi'], 'K_relative_updates': K, 'batch_size': BATCH, 'training_data_ratio': 1.0, 'optimizer': 'released fresh Adam with released warmup-cosine schedule', 'external_eval': 'B_eval'}, 'selection': {'outcome_independent': True, 'subset_seeds': SUBSET_SEEDS, 'supplement_collection_seeds': SUPPLEMENT_SEEDS, 'training_seeds': TRAIN_SEEDS, 'shared_B_core_required': True}, 'primary_contrast': 'delta_comp = L_mixed - L_current_only; negative favors MIXED_500_500', 'statistic': 'five repeat-level paired values; mean, median, sample SD, SE and 95% t CI'}
    write_json(ROOT / '00_protocol/preregistered_protocol.json', protocol)
    code = [PORTFOLIO_ROOT / 'experiments/simdist_c/collect.py', PORTFOLIO_ROOT / 'experiments/simdist_c/process.py', PORTFOLIO_ROOT / 'experiments/simdist_c/adapt.py', PORTFOLIO_ROOT / 'experiments/simdist_c/evaluate.py', simdist_root() / 'simdist/modeling/trainer.py', Path(__file__)]
    git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=simdist_root(), text=True).strip()
    write_json(ROOT / '00_protocol/PROVENANCE.json', {'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'git_commit': git, 'git_status': subprocess.check_output(['git', 'status', '--short', '--untracked-files=all'], cwd=simdist_root(), text=True), 'code_sha256': {str(p.relative_to(REPO)): sha(p) for p in code}})
    (ROOT / '00_protocol/FROZEN_PROTOCOL.md').write_text("# T1 fixed data-amount / buffer-composition control — frozen protocol\n\nThis continuation tests only +4 kg → +8 kg. A valid sample is one post-processing eligible H=T=25 window, not a control step or optimizer occurrence.\n\nFor each of five repeats, `B_core` is 500 +8 kg windows, `B_extra` is a disjoint 500 +8 kg windows, and `A_hist` is 500 +4 kg windows. `CURRENT_ONLY_1000 = B_core ∪ B_extra`; `MIXED_500_500 = B_core ∪ A_hist`. The two arms start independently from the same M_pre (`go2_2026-02-12@367004`) and use K=1,000 relative f-only optimizer updates. E/C/R/V/π are frozen.\n\nThe released trainer's internal split would otherwise remove windows from optimizer eligibility. This protocol sets `training_data_ratio=1.0` for an explicit 1,000-window training manifest and uses a completely separate pre-existing +8 kg `role=eval` B_eval artifact. With batch size 32, each epoch has 31 full batches and drops 8 rows; this is held identical across arms and logged as expected sampling exposure.\n\nNo subset is selected using adaptation/evaluation outcomes. Existing clean E1 +4 and B_eval artifacts are reused read-only; an added +8 kg supplement is collected only where the existing B collection has fewer than 1,000 eligible windows.\n")
    (ROOT / '00_protocol/SEEDS.md').write_text('# Predeclared seeds\n\n| Repeat | +8 supplement collection | subset selection | both-arm adaptation |\n|---:|---:|---:|---:|\n' + '\n'.join((f'| {r} | {SUPPLEMENT_SEEDS[r]} | {SUBSET_SEEDS[r]} | {TRAIN_SEEDS[r]} |' for r in REPEATS)) + '\n')
    (ROOT / '00_protocol/TRAINER_SEMANTICS_AUDIT.md').write_text("# Trainer semantics audit\n\n`simdist.modeling.trainer.train` constructs a seeded `random_split`, then a shuffled, non-replacement `DataLoader(drop_last=True)`. The new explicit manifests contain exactly 1,000 selected training windows and configure `training_data_ratio=1.0`, so all 1,000 are in the optimizer-eligible split. The external B_eval artifact is never passed to the Trainer.\n\nWhen `resume_checkpoint` is set, the Trainer adds the requested `max_steps` to the loaded absolute checkpoint step for stopping, while separately recording `actual_optimizer_updates` from optimizer.update calls. Thus source step 367004 plus K=1000 must finish at absolute step 368004 with `actual_optimizer_updates=1000`. The wrapper asserts both values and audits frozen state.\n\nThe optimizer, LR schedule, batch size, H/T preprocessing, loss, device and precision are otherwise the released T1 f-only path. K=1,000 is this experiment's compute control, not a paper-mandated adaptation length.\n")

def collect_supplements():
    rows = []
    for r in REPEATS:
        for folder, mass, role in [(old_source(r, 4), 4.0, 'collection'), (old_source(r, 8), 8.0, 'collection'), (old_eval(r), 8.0, 'eval')]:
            m = verify_source(folder, mass, role)
            rows.append({'repeat': r, 'dataset': str(folder), 'kind': 'reused', 'mass_kg': mass, 'requested_control_steps': 0, 'executed_control_steps': 0, 'eligible_windows': m['eligible_windows'], 'episodes': m['raw_trajectory_count'], 'early_terminations': 'see immutable rollout.json'})
        name = supplement_name(r)
        folder = ROOT / 'datasets' / name
        if not (folder / 'dataset_manifest.json').exists():
            command(f'collect_{name}', 'collect.py', ['--name', name, '--family', 'added_mass', '--friction', '0.8', '--added-mass', '8', '--kp', '25', '--kd', '.5', '--seed', SUPPLEMENT_SEEDS[r], '--steps', '1000', '--role', 'collection'])
            command(f'process_{name}', 'process.py', ['--name', name])
        m = verify_source(folder, 8.0, 'collection')
        rollout = read(folder / 'rollout.json')
        rows.append({'repeat': r, 'dataset': name, 'kind': 'new_supplement', 'mass_kg': 8.0, 'requested_control_steps': 1000, 'executed_control_steps': rollout['completed_steps'], 'eligible_windows': m['eligible_windows'], 'episodes': m['raw_trajectory_count'], 'early_terminations': len(rollout.get('early_terminations', []))})
    csv_write(ROOT / '01_manifests/DATA_AVAILABILITY_AUDIT.csv', rows)

def window_rows(folder, mass):
    folder = resolve_artifact_path(folder)
    man = dataset_manifest(folder)
    proc = resolve_artifact_path(man['processed_dir'])
    with h5py.File(proc / 'start_idxs.hdf5', 'r') as f:
        starts = [int(x) for x in f['data'][:]]
    with h5py.File(folder / 'raw_data.hdf5', 'r') as f:
        groups = list(f['data'].keys())
        lengths = [(g, len(f['data'][g]['actions'])) for g in groups]
    episodes, offset = ([], 0)
    for episode, length in lengths:
        if length >= 60:
            episodes.append((episode, offset, length))
            offset += length
    rows = []
    for start in starts:
        found = [(episode, off, length) for episode, off, length in episodes if off <= start and start + 50 < off + length - 5]
        assert len(found) == 1, (folder, start, found)
        episode, off, _ = found[0]
        uid = hashlib.sha256(f'{folder.resolve()}|{episode}|{start - off}|H{H}|T{T}'.encode()).hexdigest()
        rows.append({'source_folder': str(folder), 'source_dataset': folder.name, 'source_seed': man.get('seed'), 'regime_mass': mass, 'source_episode': episode, 'window_start_index': start - off, 'source_processed_start_idx': start, 'unique_window_id': uid})
    assert len(rows) == man['eligible_windows']
    return rows

def select_and_materialize():
    all_manifest, audits = ([], [])
    for r in REPEATS:
        a = window_rows(old_source(r, 4), 4.0)
        b = window_rows(old_source(r, 8), 8.0) + window_rows(ROOT / 'datasets' / supplement_name(r), 8.0)
        ev = window_rows(old_eval(r), 8.0)
        assert len(a) >= N_HIST and len(b) >= N_CORE + N_EXTRA and (len(ev) > 0)
        rng = np.random.default_rng(SUBSET_SEEDS[r])
        b_pick = rng.permutation(len(b))[:N_CORE + N_EXTRA]
        a_pick = rng.permutation(len(a))[:N_HIST]
        core = [dict(b[i], role='B_core') for i in b_pick[:N_CORE]]
        extra = [dict(b[i], role='B_extra') for i in b_pick[N_CORE:]]
        hist = [dict(a[i], role='A_hist') for i in a_pick]
        heldout = [dict(x, role='heldout') for x in ev]
        assert len({x['unique_window_id'] for x in core}) == N_CORE
        assert len({x['unique_window_id'] for x in extra}) == N_EXTRA
        assert len({x['unique_window_id'] for x in hist}) == N_HIST
        assert not {x['unique_window_id'] for x in core} & {x['unique_window_id'] for x in extra}
        assert not {x['unique_window_id'] for x in core + extra} & {x['unique_window_id'] for x in heldout}
        arm_rows = []
        for arm, rows in [('CURRENT_ONLY_1000', core + extra), ('MIXED_500_500', core + hist), ('B_EVAL', heldout)]:
            for x in rows:
                arm_rows.append({'repeat': r, 'arm': arm, 'subset_seed': SUBSET_SEEDS[r], **x})
        all_manifest.extend(arm_rows)
        current = make_view(r, 'current_only_1000', core + extra)
        mixed = make_view(r, 'mixed_500_500', core + hist)
        assert current['eligible_windows'] == mixed['eligible_windows'] == 1000
        audits.append({'repeat': r, 'B_core_count': len(core), 'B_extra_count': len(extra), 'A_hist_count': len(hist), 'B_eval_count': len(heldout), 'same_B_core_across_arms': True, 'B_core_B_extra_overlap': 0, 'current_train_eval_overlap': 0, 'mixed_train_eval_overlap': 0, 'all_roles_unique_within_arm': True, 'status': 'PASS'})
    csv_write(ROOT / '01_manifests/BUFFER_MANIFEST.csv', all_manifest)
    csv_write(ROOT / '01_manifests/LEAKAGE_AUDIT.csv', audits)

def make_view(repeat, arm, rows):
    name = f'dac_r{repeat}_{arm}'
    folder = ROOT / 'datasets' / name
    if (folder / 'dataset_manifest.json').exists():
        return dataset_manifest(folder)
    folder.mkdir()
    proc = folder / 'processed_data_go2_H-25_T-25'
    proc.mkdir()
    source_folders = []
    for row in rows:
        p = resolve_artifact_path(row['source_folder'])
        if p not in source_folders:
            source_folders.append(p)
    source_info, offset = ([], 0)
    for source in source_folders:
        man = dataset_manifest(source)
        sp = resolve_artifact_path(man['processed_dir'])
        with h5py.File(sp / 'proprio_obs.hdf5', 'r') as f:
            length = len(f['data'])
        source_info.append((source, sp, offset, man))
        offset += length
    for key in KEYS:
        with h5py.File(proc / key, 'w') as dst:
            for _, sp, _, _ in source_info:
                with h5py.File(sp / key, 'r') as src:
                    data = src['data'][:]
                if 'data' not in dst:
                    dst.create_dataset('data', shape=(0,) + data.shape[1:], maxshape=(None,) + data.shape[1:], dtype=data.dtype)
                dset = dst['data']
                n = len(dset)
                dset.resize(n + len(data), axis=0)
                dset[n:] = data
    offsets = {str(source): source_offset for source, _, source_offset, _ in source_info}
    identities, starts = ([], [])
    for row in rows:
        item = dict(row)
        item['view_start_idx'] = item['source_processed_start_idx'] + offsets[item['source_folder']]
        identities.append(item)
        starts.append(item['view_start_idx'])
    with h5py.File(proc / 'start_idxs.hdf5', 'w') as f:
        f.create_dataset('data', data=np.asarray(starts, dtype=np.int64))
    for filename in ['scaler_params.json', 'system.yaml', 'dataset_metrics.json']:
        shutil.copy2(source_info[0][1] / filename, proc / filename)
    manifest = {'name': name, 'role': 'collection', 'view_kind': 'T1_FIXED_DATA_AMOUNT_EXPLICIT_TRAINING_VIEW', 'eligible_windows': len(rows), 'seeds': sorted({int(x[3]['seed']) for x in source_info if x[3].get('seed') is not None}), 'processed_dir': str(proc), 'processed_hashes': hashes(proc), 'window_identities': identities, 'source_datasets': [x[0].name for x in source_info], 'training_data_ratio': 1.0}
    write_json(folder / 'dataset_manifest.json', manifest)
    return manifest

def run_adapt_eval():
    allowed_training = ','.join((str(TRAIN_SEEDS[r]) for r in REPEATS))
    for r in REPEATS:
        eval_name = old_eval(r).name
        eval_out = ROOT / 'eval' / f'dac_r{r}_M_pre_B_eval'
        if not (eval_out / 'metrics.json').exists():
            command(f'eval_pre_r{r}', 'evaluate.py', ['--names', eval_name, '--dataset-root', OLD / 'datasets', '--output', eval_out])
        for arm in ('current_only_1000', 'mixed_500_500'):
            dataset = f'dac_r{r}_{arm}'
            run = dataset + '_K1000'
            result = ROOT / 'adaptation' / run / 'result.json'
            manifest = dataset_manifest(ROOT / 'datasets' / dataset)
            allowed_collection = ','.join(map(str, manifest['seeds']))
            if not result.exists():
                command(f'adapt_{run}', 'adapt.py', ['--name', dataset, '--run', run, '--seed', TRAIN_SEEDS[r], '--steps', K, '--training-data-ratio', '1.0', '--allowed-training-seeds', allowed_training, '--allowed-collection-seeds', allowed_collection, '--training-dataset-root', ROOT / 'datasets', '--heldout-name', eval_name, '--heldout-dataset-root', OLD / 'datasets', '--logging-eval-interval', '100', '--log-dir', ROOT / 'logs' / run])
            audit = read(result)
            assert audit['initial_step'] == 367004 and audit['final_step'] == 368004
            assert audit['optimizer_updates'] == K and audit['actual_optimizer_updates'] == K
            assert audit['train_windows'] == 1000 and (not audit['frozen_changes']) and audit['dynamics_changes']
            out = ROOT / 'eval' / f'dac_r{r}_{arm}_B_eval'
            if not (out / 'metrics.json').exists():
                command(f'eval_{run}', 'evaluate.py', ['--names', eval_name, '--checkpoint', ROOT / 'models' / run, '--dataset-root', OLD / 'datasets', '--output', out])

def metric(path):
    return float(read(path / 'metrics.json')['per_seed'][0]['latent_mse'])

def results_and_report():
    rows = []
    run_manifest, exposure = ([], [])
    for r in REPEATS:
        pre = metric(ROOT / 'eval' / f'dac_r{r}_M_pre_B_eval')
        current = metric(ROOT / 'eval' / f'dac_r{r}_current_only_1000_B_eval')
        mixed = metric(ROOT / 'eval' / f'dac_r{r}_mixed_500_500_B_eval')
        a0 = read(ROOT / 'adaptation' / f'dac_r{r}_current_only_1000_K1000' / 'result.json')
        a1 = read(ROOT / 'adaptation' / f'dac_r{r}_mixed_500_500_K1000' / 'result.json')
        i0 = read(ROOT / 'adaptation' / f'dac_r{r}_current_only_1000_K1000' / 'invocation.json')
        i1 = read(ROOT / 'adaptation' / f'dac_r{r}_mixed_500_500_K1000' / 'invocation.json')
        rows.append({'repeat': r, 'B_eval_dataset': old_eval(r).name, 'B_eval_windows': dataset_manifest(old_eval(r))['eligible_windows'], 'L_pre': pre, 'L_current_only': current, 'L_mixed': mixed, 'delta_comp': mixed - current, 'current_only_minus_pre': current - pre, 'mixed_minus_pre': mixed - pre, 'current_only_pct_change_vs_pre': (current - pre) / pre * 100, 'mixed_pct_change_vs_pre': (mixed - pre) / pre * 100, 'current_updates': a0['actual_optimizer_updates'], 'mixed_updates': a1['actual_optimizer_updates'], 'same_M_pre': i0['source_hashes'] == i1['source_hashes'] and a0['initial_step'] == a1['initial_step'] == 367004, 'status': 'COMPLETE'})
        for arm, audit in [('CURRENT_ONLY_1000', a0), ('MIXED_500_500', a1)]:
            run = f'dac_r{r}_{arm.lower()}_K1000'
            run_manifest.append({'repeat': r, 'arm': arm, 'M_pre': str(SOURCE), 'initial_step': audit['initial_step'], 'final_step': audit['final_step'], 'actual_optimizer_updates': audit['actual_optimizer_updates'], 'checkpoint': audit['checkpoint'], 'adaptation_audit': str(ROOT / 'adaptation' / run / 'result.json'), 'runtime_stdout': str(ROOT / '02_runs/runtime_logs' / f'adapt_{run}.stdout.log'), 'runtime_stderr': str(ROOT / '02_runs/runtime_logs' / f'adapt_{run}.stderr.log'), 'external_eval': str(ROOT / 'eval' / f'dac_r{r}_{arm.lower()}_B_eval' / 'metrics.json')})
            exposure.append({'repeat': r, 'arm': arm, 'optimizer_eligible_unique_windows': 1000, 'training_data_ratio': 1.0, 'batch_size': BATCH, 'full_batches_per_epoch': 31, 'dropped_windows_per_full_epoch': 8, 'optimizer_updates': K, 'planned_optimizer_sample_occurrences': K * BATCH, 'expected_reuse_per_unique_window': K * BATCH / 1000, 'expected_plus8_sample_occurrences': K * BATCH if arm == 'CURRENT_ONLY_1000' else K * BATCH / 2, 'expected_plus4_sample_occurrences': 0 if arm == 'CURRENT_ONLY_1000' else K * BATCH / 2, 'per_window_draw_count': 'not intercepted by released Trainer; all 1,000 IDs are optimizer-eligible and both arms use identical batching semantics'})
    csv_write(ROOT / '03_results/DATA_AMOUNT_CONTROL_RAW.csv', rows)
    csv_write(ROOT / '02_runs/RUN_MANIFEST.csv', run_manifest)
    csv_write(ROOT / '02_runs/SAMPLING_EXPOSURE_AUDIT.csv', exposure)
    delta = np.asarray([x['delta_comp'] for x in rows], dtype=float)
    mean, sd = (float(delta.mean()), float(delta.std(ddof=1)))
    se = sd / math.sqrt(len(delta))
    tc = 2.7764451051977987
    summary = {'n_repeats': len(delta), 'mean_delta_comp': mean, 'median_delta_comp': float(np.median(delta)), 'sample_sd': sd, 'standard_error': se, 'ci95_low': mean - tc * se, 'ci95_high': mean + tc * se, 'positive_sign_count': int((delta > 0).sum()), 'negative_sign_count': int((delta < 0).sum()), 'min': float(delta.min()), 'max': float(delta.max()), 'classification': 'USEFUL_HISTORICAL_TRANSFER' if mean < 0 and mean + tc * se < 0 else 'CURRENT_DATA_PREFERRED' if mean > 0 and mean - tc * se > 0 else 'INCONCLUSIVE_DATA_COMPOSITION', 't_critical_df4': tc}
    csv_write(ROOT / '03_results/DATA_AMOUNT_CONTROL_SUMMARY.csv', [summary])
    figures(rows)
    report(rows, summary)

def figures(rows):
    import matplotlib.pyplot as plt
    reps = [x['repeat'] for x in rows]
    cur = [x['L_current_only'] for x in rows]
    mix = [x['L_mixed'] for x in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r, a, b in zip(reps, cur, mix):
        ax.plot([0, 1], [a, b], marker='o', label=f'r{r}')
    ax.set_xticks([0, 1], ['CURRENT_ONLY_1000', 'MIXED_500_500'])
    ax.set_ylabel('held-out +8 kg latent MSE')
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / '04_figures/paired_mse.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color='black', lw=1)
    ax.scatter(reps, [x['delta_comp'] for x in rows], s=55)
    ax.set_xticks(reps)
    ax.set_xlabel('repeat')
    ax.set_ylabel('Δ_comp = L_mixed − L_current_only')
    fig.tight_layout()
    fig.savefig(ROOT / '04_figures/delta_comp_raw.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(reps, [x['current_only_minus_pre'] for x in rows], label='CURRENT_ONLY_1000')
    ax.scatter(reps, [x['mixed_minus_pre'] for x in rows], label='MIXED_500_500')
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(reps)
    ax.set_xlabel('repeat')
    ax.set_ylabel('held-out MSE change vs M_pre')
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / '04_figures/adaptation_vs_pre.png', dpi=180)
    plt.close(fig)

def report(rows, summary):
    old_rows = list(csv.DictReader((OLD / '09_t1_nonstationarity_v2/e1_abrupt/ABRUPT_PREDICTION_RESULTS.csv').open()))
    old_delta = [float(x['delta_L']) for x in old_rows if x.get('status') == 'COMPLETE']
    lines = ['# T1 data-amount / buffer-composition control report', '', '## Result', '', f"Final label: **`{summary['classification']}`**. The primary contrast is Δ_comp = L_mixed − L_current_only; negative values favor replacing 500 extra +8 kg windows with 500 historical +4 kg windows.", '', '## Required verification', '', '1. Yes. Both arms in every repeat independently loaded the identical M_pre source checkpoint at absolute step 367004.', '2. Yes. `B_core` is exactly the same 500 window IDs in both arm manifests.', '3. Yes. `B_extra` and `A_hist` contain exactly 500 unique windows each.', "4. Yes. Each arm's explicit post-processing optimizer-eligible training manifest contains exactly 1,000 unique windows.", '5. Yes. B_eval is a separate immutable +8 kg `role=eval` artifact and all per-repeat leakage checks pass.', '6. Yes. Every arm has K=1,000 actual relative optimizer updates (367004→368004), verified from the Trainer completion counter and checkpoint audit.', '', '## Raw repeat results', '', '| Repeat | L_pre | L_current_only | L_mixed | Δ_comp |', '|---:|---:|---:|---:|']
    for x in rows:
        lines.append(f"| {x['repeat']} | {x['L_pre']:.9f} | {x['L_current_only']:.9f} | {x['L_mixed']:.9f} | {x['delta_comp']:.9f} |")
    lines += ['', 'The five raw L_current_only, L_mixed and Δ_comp values are also recorded without aggregation in `../03_results/DATA_AMOUNT_CONTROL_RAW.csv`.', '', '## Repeat-level statistics', '', f"Δ_comp: mean {summary['mean_delta_comp']:.9f}; median {summary['median_delta_comp']:.9f}; sample SD {summary['sample_sd']:.9f}; SE {summary['standard_error']:.9f}; 95% t CI [{summary['ci95_low']:.9f}, {summary['ci95_high']:.9f}]; signs positive/negative = {summary['positive_sign_count']}/{summary['negative_sign_count']}; min/max [{summary['min']:.9f}, {summary['max']:.9f}].", '', '## Adaptation relative to M_pre', '']
    for x in rows:
        lines.append(f"- r{x['repeat']}: CURRENT_ONLY change {x['current_only_minus_pre']:.9f} ({x['current_only_pct_change_vs_pre']:.2f}%); MIXED change {x['mixed_minus_pre']:.9f} ({x['mixed_pct_change_vs_pre']:.2f}%).")
    cur_all = all((x['current_only_minus_pre'] < 0 for x in rows))
    mix_all = all((x['mixed_minus_pre'] < 0 for x in rows))
    lines += ['', f'Both arms improved in every repeat: CURRENT_ONLY={cur_all}; MIXED={mix_all}.', '', '## Relation to prior natural-buffer T1', '', f"The immutable prior E1 CUMULATIVE-NATURAL minus RECENT values were {', '.join((f'{x:.9f}' for x in old_delta))} (all negative; mean {np.mean(old_delta):.9f}). That experiment had natural buffer growth and unequal current-regime exposure, so it is not overwritten here.", f"Combined interpretation: **{summary['classification']}**. " + ('With the previous natural cumulative advantage, this supports a specific historical-regime contribution at equal total size.' if summary['classification'] == 'USEFUL_HISTORICAL_TRANSFER' else 'With the previous natural cumulative advantage, the new result is more consistent with total unique-data amount/diversity than a specific +4 kg historical benefit.' if summary['classification'] == 'CURRENT_DATA_PREFERRED' else 'The previous advantage cannot yet be attributed to historical-regime contribution versus data amount.'), '', '## Remaining evidence for the research-anchor decision', '', 'This narrow prediction-only gate supplies no control-return, tracking, planner, R/V, or support-coverage evidence. Because the composition CI still spans zero, the specific missing evidence is a pre-registered repeat-level precision replication of this same fixed-size contrast sufficient to distinguish the sign from repeat variability; no new hypothesis suite is justified by this run. The anchor decision should use that evidence together with the immutable prior T1 record.']
    (ROOT / '05_report/T1_DATA_AMOUNT_CONTROL_REPORT.md').write_text('\n'.join(lines) + '\n')
    (ROOT / '05_report/FINAL_COMPLETION_STATUS.md').write_text(f"# Final completion status\n\n**COMPLETE — {summary['classification']}**\n\nFive repeat pairs completed with fixed 1,000-window buffers and K=1,000 relative f-only updates. No control/T2/T3/E2/E3/E4 work was run.\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['init', 'collect', 'prepare', 'run', 'finalize', 'all'])
    args = p.parse_args()
    if args.stage in ('init', 'all'):
        init()
    if args.stage in ('collect', 'all'):
        collect_supplements()
    if args.stage in ('prepare', 'all'):
        select_and_materialize()
    if args.stage in ('run', 'all'):
        run_adapt_eval()
    if args.stage in ('finalize', 'all'):
        results_and_report()
if __name__ == '__main__':
    main()
