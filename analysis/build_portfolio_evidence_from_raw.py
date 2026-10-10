"""Reaggregate saved SimDist measurements; never import an experiment or run a model.

Each CSV has an independent publication gate. An unavailable historical test
is an explicit NOT_REPRODUCED status row with no p value. See the adjacent
PROVENANCE document for schemas, selection rules and outstanding evidence.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import io
import json
import math
import os
import re
import shutil
import statistics
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
OLD = set(range(23501, 23601))
NEW = set(range(24001, 24101))
EVAL = OLD | NEW
HELDOUT = set(range(186001, 186021))
MASS = ['m0', 'old_chain1', 'old_chain2', 'new_chain3', 'new_chain4', 'old_DP', 'new_DP2']
MODELS = {
    'mass8': MASS,
    'friction015': ['m0', 'friction_chain1', 'friction_chain2', 'friction_60k', 'friction_90k'],
    'gain070': ['m0', 'gain_chain1', 'gain_chain2'],
    'flat': ['m0', 'flat075_chain1', 'flat075_chain2'],
}
FILES = ['mass_chains_200seed.csv', 'chain_equality_test.csv', 'value_mc_rank.csv',
         'common_state_dynamics_error.csv', 'target_failure_summary.csv',
         'flat_velocity_comparison.csv']


class EvidenceError(ValueError):
    """A sanitized, public-safe validation failure."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def finite(value, label):
    result = float(value)
    require(math.isfinite(result), 'nonfinite: ' + label)
    return result


def integer(value, label):
    result = finite(value, label)
    require(result == int(result), 'noninteger: ' + label)
    return int(result)


def flag(value, label):
    result = integer(value, label)
    require(result in (0, 1), 'nonbinary: ' + label)
    return bool(result)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Raw:
    def __init__(self, root):
        self.root = root.resolve(strict=True)
        self.inputs = {}
        self.stats = {}
        self.checkpoints = {}
        self.checkpoint_stats = {}
        self.bindings = {}

    def path(self, relative):
        p = self.root / relative
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                'unsafe raw-relative path')
        require(p.resolve().is_relative_to(self.root), 'raw input escapes root')
        require(p.is_file(), 'missing input: ' + relative)
        return p

    def register(self, relative, basis):
        p = self.path(relative)
        current = (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
        if relative not in self.inputs:
            sha = digest(p)
            require(current == (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino),
                    'input changed while hashing: ' + relative)
            self.stats[relative] = current
            self.inputs[relative] = {'path': relative, 'sha256': sha,
                                     'size_bytes': current[0], 'basis': basis}
        else:
            require(current == self.stats[relative], 'input changed: ' + relative)
        return p

    def json(self, relative, basis='seed/state-level measurement'):
        return json.loads(self.register(relative, basis).read_text())

    def csv(self, relative, basis='seed/episode-level measurement'):
        with self.register(relative, basis).open(newline='') as f:
            reader = csv.DictReader(f)
            require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames),
                    'missing/duplicate headers: ' + relative)
            rows = list(reader)
        require(all(None not in row and None not in row.values() for row in rows),
                'malformed CSV: ' + relative)
        return rows

    def reference(self, original):
        """Normalize historical metadata without emitting machine-specific prefixes."""
        parts = Path(original).parts
        marker = 'scratch'
        require(marker in parts, 'unrecognized historical raw reference')
        relative = Path(*parts[parts.index(marker) + 1:]).as_posix()
        self.path(relative)
        return relative

    def unchanged(self):
        for relative, previous in self.stats.items():
            p = self.path(relative)
            require(previous == (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino),
                    'input changed during analysis: ' + relative)
        for p, previous in self.checkpoint_stats.items():
            require(previous == (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino),
                    'checkpoint changed during analysis')

    def checkpoint(self, original, identifier):
        parts=Path(original).parts
        if 'scratch' in parts:
            p=(self.root/Path(*parts[parts.index('scratch')+1:])).resolve()
        else:
            require('checkpoints' in parts,'unrecognized checkpoint reference')
            p=(self.root.parent/Path(*parts[parts.index('checkpoints'):])).resolve()
        require(p.is_relative_to(self.root.parent) and p.is_dir(), 'checkpoint binding outside original project or missing')
        relative = p.relative_to(self.root.parent).as_posix()
        if relative not in self.checkpoints:
            files = []
            for directory, dirs, names in os.walk(p, followlinks=False):
                require(not any((Path(directory)/name).is_symlink() for name in dirs + names), 'checkpoint contains symlink')
                for name in sorted(names):
                    q = Path(directory)/name
                    before = (q.stat().st_size, q.stat().st_mtime_ns, q.stat().st_ino)
                    sha = digest(q)
                    require(before == (q.stat().st_size, q.stat().st_mtime_ns, q.stat().st_ino), 'checkpoint changed while hashing')
                    self.checkpoint_stats[q] = before
                    files.append({'path': q.relative_to(p).as_posix(), 'sha256': sha, 'size_bytes': before[0]})
            steps = [int(q.name) for q in p.iterdir() if q.is_dir() and q.name.isdecimal()]
            require(len(steps) == 1 and files, 'ambiguous checkpoint step')
            lines = ''.join(x['path']+'\t'+x['sha256']+'\n' for x in sorted(files,key=lambda x:x['path']))
            self.checkpoints[relative] = {'step': steps[0], 'sha256_tree': hashlib.sha256(lines.encode()).hexdigest(),
                                         'files': files, 'ids': []}
        result = self.checkpoints[relative]
        if identifier not in result['ids']:
            result['ids'].append(identifier)
        return {'checkpoint_id': identifier, 'checkpoint_step': result['step'], 'checkpoint_sha256_tree': result['sha256_tree']}


def selection(directory, seeds):
    return directory, seeds


def pinned_config(raw, relative):
    """Only the discovered legacy schema may use a historical configuration pin."""
    pin = 'positive_control_variance_v1/source_hashes_before.json'
    recorded = raw.json(pin, 'historical source-hash metadata; no result numbers')
    hashes = [sha for path, sha in recorded.items() if path.endswith('/scratch/' + relative)]
    raw.register(relative, 'historically pinned producer configuration')
    require(len(hashes) == 1 and hashes[0] == raw.inputs[relative]['sha256'],
            'historical producer pin mismatch: ' + relative)
    text = raw.path(relative).read_text()
    for binding in ['tr.sub_terrains = {"boxes": tr.sub_terrains["boxes"]}',
                    'mat["static_friction_range"] = (0.8, 0.8)',
                    'mat["dynamic_friction_range"] = (0.8, 0.8)']:
        require(binding in text, 'historical material/terrain binding changed: ' + relative)


def episode_sources(target, model):
    """Discovered physical directories; aggregates are never read by the builder."""
    prefix = 'multi_gap_200seed_extension_v1/results/evaluations/'
    if target == 'mass8':
        historic = {
            'm0': [selection('results/s1_v2/runs', set(range(23501, 23511))),
                   selection('adaptation_curve_v1/results/evaluations/baseline_target', set(range(23511, 23521))),
                   selection('score_probe_v1/results/evaluations/m0', set(range(23521, 23601)))],
            'old_chain1': [selection('adaptation_curve_v1/results/evaluations/stage_30000', set(range(23501, 23521))),
                           selection('score_probe_v1/results/evaluations/cp30000', set(range(23521, 23601)))],
            'old_chain2': [selection('followup_v1/results/chain/evaluation', OLD)],
            'old_DP': [selection('module_combo_v1/results/followup_repro/mass8_d_p/evaluations/d_pnone_mppi', OLD)],
        }
        old = historic.get(model, [selection('positive_control_variance_v1/results/mass8/evaluations/' + model, OLD)])
        new = ('chain_variance_recheck_v1/results/evaluations/mass8_new_' + model
               if model in ['m0', 'old_DP', 'new_DP2'] else prefix + 'mass8_' + model)
        return old + [selection(new, NEW)]
    if target in ('friction015', 'gain070'):
        if model.endswith(('60k', '90k')):
            return [selection(prefix + target + '_' + model.split('_')[-1], EVAL)]
        family = 'friction' if target == 'friction015' else 'gain'
        second = 'module_combo_v1/results/followup_repro/' + target + '/second_chain/evaluations/'
        if model == 'm0':
            old = [selection('module_combo_v1/results/' + target + '/evaluations/n0p0_mppi', set(range(23501, 23551))),
                   selection(second + 'm0_extension', set(range(23551, 23601)))]
        elif model == family + '_chain1':
            old = [selection('module_combo_v1/results/' + target + '/evaluations/d30p0_mppi', set(range(23501, 23551))),
                   selection(second + 'chain1_30k_extension', set(range(23551, 23601)))]
        else:
            old = [selection(second + 'chain2_30k', OLD)]
        return old + [selection(prefix + target + '_' + model, NEW)]
    if model.startswith('flat075_'):
        return [selection('flat_velocity_matched_v1/results/evaluations/v075/' + model, EVAL)]
    return [selection(prefix + 'flat075_' + model, EVAL)]


def load_episodes(raw, target, model):
    episodes = {}
    for directory, selected in episode_sources(target, model):
        seen = set()
        pattern = 'mppi_mass8_v10_seed*/summary.json' if directory == 'results/s1_v2/runs' else 'mppi_*_seed*/summary.json'
        for p in sorted((raw.root / directory).glob(pattern)):
            relative = p.relative_to(raw.root).as_posix()
            s = raw.json(relative)
            seed = integer(s['seed'], 'seed')
            if seed not in selected:
                continue
            require(seed not in episodes, f'duplicate evaluation seed: {target}/{model}/{seed}')
            seen.add(seed)
            velocity = .75 if target == 'flat' else 1.
            require(s['status'] in ('evaluated', 'warmup_failed'), 'unfinished episode: ' + relative)
            require(integer(s['model_step'],'evaluation checkpoint step')==raw.bindings[target,model]['checkpoint_step'],
                    'evaluation/checkpoint source binding mismatch: '+relative)
            require(s['controller'] == 'mppi' and s['forward_target_mps'] == velocity,
                    'controller/velocity mismatch: ' + relative)
            if 'terrain' in s:
                require(s['terrain'] == ('plane' if target == 'flat' else 'boxes'),
                        'terrain mismatch: ' + relative)
            else:
                require(target != 'flat', 'missing flat terrain condition: ' + relative)
                # Older evaluators omit terrain; their pinned configuration fixes boxes.
                pinned_config(raw, 'go2_probe_common.py')
            expected = {'mass8': (.8, 8., 25., .5), 'friction015': (.15, 0., 25., .5),
                        'gain070': (.8, 0., 17.5, .35), 'flat': (.10, 0., 25., .5)}[target]
            physical = s['physical_readback']
            friction = s.get('foot_friction', physical.get('foot_friction'))
            if friction is None:
                require(target == 'mass8', 'missing friction readback: ' + relative)
                pinned_config(raw, 'go2_probe_common.py')
                friction = .8  # Explicit historical schema: pinned material config, not a column guess.
            for value, wanted in zip([friction, s['mass_added_kg'], physical['kp'], physical['kd']], expected):
                require(abs(finite(value, 'physical condition') - wanted) < 1e-5,
                        'physical condition mismatch: ' + relative)
            require(s['joint_reset_max_error'] < 1e-5 and s['joint_velocity_reset_max_error'] < 1e-5,
                    'reset error: ' + relative)
            steps = raw.csv(relative.replace('summary.json', 'evaluation_steps.csv')) if s['episode_length'] else []
            warm = raw.csv(relative.replace('summary.json', 'warmup_steps.csv'))
            n = integer(s['episode_length'], 'episode_length')
            require(0 <= n <= 1000 and len(steps) == n and len(warm) == s['warmup_steps'],
                    'episode length mismatch: ' + relative)
            require([integer(x['step'], 'step') for x in steps] == list(range(1, n + 1)),
                    'step sequence mismatch: ' + relative)
            failure = flag(s['fall'], 'fall') or s['status'] == 'warmup_failed'
            terminated = [flag(x['terminated'], 'terminated') for x in steps]
            require(sum(terminated) == int(bool(steps) and failure) and
                    (not any(terminated) or terminated[-1]), 'termination mismatch: ' + relative)
            require(failure == (s['status'] == 'warmup_failed' or bool(terminated and terminated[-1])),
                    'failure mismatch: ' + relative)
            require(n == 1000 or failure, 'incomplete nonfailure episode: ' + relative)
            require(abs(sum(finite(x['reward'], 'reward') for x in steps) - finite(s['return'], 'return')) < 1e-7,
                    'return mismatch: ' + relative)
            require(s['status'] != 'evaluated' or s['warmup_steps'] == 25, 'warmup mismatch: ' + relative)
            tile = None
            if target != 'flat':
                require(s['warmup_steps'] == 25 and s['status'] == 'evaluated',
                        'tile clock undefined after warmup failure: ' + relative)
                clock = np.zeros(1101, dtype=np.float32)
                for i in range(1100):
                    clock[i + 1] = np.float32(clock[i] + np.float32(.02))
                position = [float(clock[25 + integer(x['step'], 'step') - 1]) - finite(x['command_vx'], 'command_vx') for x in steps]
                exited = any(x > 7.425 for x in position)
                require(failure or exited, 'right-censored tile outcome: ' + relative)
                tile = int(failure and not exited)
            episodes[seed] = {'failure': int(failure), 'tile_failure': tile,
                              'source': relative, 'n_steps': n}
        require(seen == selected, f'incomplete selected seed block: {directory}')
    require(set(episodes) == EVAL, f'200-seed coverage mismatch: {target}/{model}')
    return episodes


def loss_csv(raw, relative, dataset, models, seeds):
    rows = raw.csv(relative, 'per-rollout held-out prediction MSE; not a report aggregate')
    required = {'model', 'checkpoint', 'checkpoint_step', 'dataset', 'rollout', 'windows',
                'one_step_latent_mse', 'full_25_step_latent_mse'}
    require(rows and required <= rows[0].keys(), 'held-out schema mismatch: ' + relative)
    result = {}
    references = None
    for model in models:
        selected = [x for x in rows if x['model'] == model and x['dataset'] == dataset]
        by = {}
        checkpoint = None
        for row in selected:
            binding = raw.checkpoint(row['checkpoint'], 'loss:' + dataset + ':' + relative.split('/')[0] + ':' + model)
            require(binding['checkpoint_step'] == integer(row['checkpoint_step'], 'checkpoint step'), 'loss checkpoint step mismatch')
            require(checkpoint is None or checkpoint == binding, 'mixed checkpoints in seed loss rows')
            checkpoint = binding
            ref = raw.reference(row['rollout'])
            raw.register(ref, 'underlying held-out collection; never copied or evaluated')
            match = re.search(r'/seed_(\d+)/rollout\.h5$', '/' + ref)
            require(match is not None, 'unrecognized held-out seed path: ' + relative)
            seed = int(match.group(1))
            require(seed not in by, 'duplicate held-out seed: ' + relative)
            n = integer(row['windows'], 'windows')
            value = finite(row['full_25_step_latent_mse'], 'latent MSE')
            require(n > 0 and value >= 0 and finite(row['one_step_latent_mse'], 'one-step MSE') >= 0,
                    'invalid held-out measurement: ' + relative)
            by[seed] = (value, n, ref)
        require(set(by) == seeds, 'held-out seed coverage mismatch: ' + relative + '/' + model)
        current_refs = {seed: (x[1], x[2]) for seed, x in by.items()}
        require(references is None or current_refs == references,
                'models use different validation data/windows: ' + relative)
        references = current_refs
        values = [x[0] for x in by.values()]
        result[model] = {'loss25': statistics.mean(values), 'loss25_sd': statistics.stdev(values),
                         'loss_n': len(values), 'windows': sum(x[1] for x in by.values()),
                         'loss25_window_pooled': sum(x[0]*x[1] for x in by.values()) / sum(x[1] for x in by.values()),
                         'loss_basis': 'arithmetic mean of independent seed MSEs', 'loss_source': relative,
                         'loss_sha256': raw.inputs[relative]['sha256'], 'dataset': dataset,
                         'loss_seed_ranges':f'{min(seeds)}-{max(seeds)}'}
        result[model].update(checkpoint)
    return result


def common_losses(raw, target, models):
    result = {}
    shared = None
    for model in models:
        values, windows, refs = [], {}, []
        directory = f'common_state_gap_v1/results/loss/{target}/{model}'
        for seed in sorted(HELDOUT):
            collection = f'common_state_gap_v1/results/collection/{target}/seed_{seed}/rollout.h5'
            raw.register(collection, 'shared held-out collection; no inference')
            ref = f'{directory}/seed_{seed}/summary.json'
            s = raw.json(ref)
            require((s['target'], s['model'], s['seed']) == (target, model, seed), 'held-out identity mismatch: ' + ref)
            value = finite(s['loss25'], 'common loss25')
            n = integer(s['windows'], 'windows')
            require(value >= 0 and n > 0, 'invalid common loss: ' + ref)
            windows[seed] = n
            values.append(value)
            refs.append(ref)
        require(shared is None or windows == shared, 'common validation windows differ: ' + target)
        shared = windows
        result[model] = {'loss25': statistics.mean(values), 'loss25_sd': statistics.stdev(values),
                         'loss_n': 20, 'windows': sum(windows.values()),
                         'loss_basis': 'arithmetic mean of 20 common-state collection seed MSEs',
                         'loss_source': directory + '/seed_*/summary.json',
                         'loss_sha256': source_set_hash(raw, refs), 'dataset': 'common075' if target == 'flat' else 'common100',
                         'loss_seed_ranges':'186001-186020',
                         'loss25_window_pooled':sum(v*n for v,n in zip(values,windows.values()))/sum(windows.values())}
        result[model].update(raw.bindings[target, model])
    return result


def source_set_hash(raw, references):
    text = ''.join(p + '\t' + raw.inputs[p]['sha256'] + '\n' for p in sorted(references))
    return hashlib.sha256(text.encode()).hexdigest()


def row_sources(raw, episodes):
    summaries = [s['source'] for s in episodes.values()]
    refs = [ref for summary in summaries for ref in
            [summary, summary.replace('summary.json', 'evaluation_steps.csv'),
             summary.replace('summary.json', 'warmup_steps.csv')] if ref in raw.inputs]
    return {'source': ' | '.join(sorted({str(Path(p).parent.parent) for p in summaries})),
            'source_sha256_set': source_set_hash(raw, refs),
            'seed_ranges': '23501-23600;24001-24100'}


def count_row(raw, target, model, episodes, basis):
    failures = sum(x['tile_failure'] if basis == 'tile' else x['failure'] for x in episodes.values())
    return {'target': target, 'model': model, 'basis': 'pre-action displacement >7.425m first-exit' if basis == 'tile' else 'full episode termination or warmup failure; max1000',
            'n_eval': 200, 'failures': failures, 'failure_pct': failures / 2,
            'eval_command_mps': .75 if target == 'flat' else 1.,
            'evaluation_checkpoint': raw.bindings[target, model], **row_sources(raw, episodes)}


def value_mc(raw):
    roots = [('source', 'candidate_rank_v1/results/full/m0_mass0', 175),
             ('mass8', 'candidate_rank_v1/results/full/m0_mass8', 175),
             ('friction015', 'gap_diagnosis_v1/results/branches/friction_m0', 200),
             ('gain070', 'gap_diagnosis_v1/results/branches/gain_m0', 200)]
    output = []
    for target, directory, horizon in roots:
        pairs, seen, refs, observed = [], set(), [], []
        paths = sorted((raw.root / directory).glob('seed_*/snapshot_*/candidates.csv'))
        require(paths, 'missing expert rollout family: ' + directory)
        excluded = 0
        for p in paths:
            ref = p.relative_to(raw.root).as_posix()
            rows = raw.csv(ref, 'snapshot-level candidate/privileged critic measurement')
            ids = [i for i, row in enumerate(rows) if row['candidate_kind'] == 'expert_closed']
            require(len(ids) == 1, 'expert branch multiplicity: ' + ref)
            i = ids[0]
            identity = (integer(rows[i]['seed'], 'seed'), integer(rows[i]['snapshot_step'], 'snapshot step'))
            require(identity not in seen, 'duplicate expert snapshot: ' + ref)
            seen.add(identity)
            arrays = p.with_name('sequences_with_plans.npz')
            if not arrays.exists():
                arrays = p.with_name('sequences.npz')
            ar = arrays.relative_to(raw.root).as_posix()
            raw.register(ar, 'saved expert reward/value-head sequences')
            refs.extend([ref, ar])
            with np.load(arrays, allow_pickle=False) as z:
                require(list(z['candidate_kinds']) == [x['candidate_kind'] for x in rows], 'candidate order mismatch: ' + ref)
                require(z['actual_env_reward_sequence'].shape[1]==25+horizon,'saved MC storage horizon mismatch: '+ref)
                if not bool(z['actual_head_valid'][i]):
                    excluded += 1
                    continue
                n = integer(rows[i]['actual_steps'], 'actual steps')
                require(25 <= n <= 25 + horizon, 'expert horizon mismatch: ' + ref)
                rewards = z['actual_env_reward_sequence'][i, 25:n].astype(float)
                head = finite(z['actual_latent_value_head_sequence'][i, -1], 'value head')
                critic = finite(rows[i]['expert_critic_step25'], 'expert critic')
                require(len(rewards) == n - 25 and np.isfinite(rewards).all(), 'MC rewards missing: ' + ref)
                mc = float(np.sum(rewards * .99**np.arange(len(rewards))))
                pairs.append((head, critic, mc))
                observed.append(n - 25)
        require(len(pairs) > 2, 'insufficient value/MC pairs: ' + directory)
        a = np.asarray(pairs)
        head_rho = finite(spearmanr(a[:, 0], a[:, 2]).statistic, 'head Spearman')
        critic_rho = finite(spearmanr(a[:, 1], a[:, 2]).statistic, 'critic Spearman')
        output.append({'target': target, 'model': 'm0', 'rho_head_mc': head_rho, 'rho_critic_mc': critic_rho,
                       'n_eval': len(pairs), 'n_selected': len(paths), 'n_excluded_incomplete25': excluded,
                       'mc_horizon': horizon, 'mc_observed_min': min(observed), 'mc_observed_max': max(observed),
                       'discount': .99, 'basis': 'expert branch head/critic at step25 versus subsequent discounted env return; pooled snapshots',
                       'source': directory, 'source_sha256_set': source_set_hash(raw, refs)})
    return output


def common_dynamics(raw):
    output = []
    groups = {'mass8': MASS, 'friction015': MODELS['friction015'],
              'gain070': MODELS['gain070'], 'flat': ['m0', 'flat_chain1', 'flat_chain2']}
    for target, models in groups.items():
        shared = None
        model_stats = {}
        all_refs = []
        for model in models:
            selected, values = {}, []
            for seed in sorted(HELDOUT):
                ref = f'common_state_gap_v1/results/loss/{target}/{model}/seed_{seed}/states.json'
                states = raw.json(ref, 'state-level selection and saved-state identity')
                all_refs.append(ref)
                for s in states:
                    key = (seed, s['state_id'])
                    require(s['seed'] == seed and key not in selected, 'duplicate/mismatched shared state: ' + ref)
                    state = raw.reference(s['state_path'])
                    selected[key] = (state, s['step'], s['episode'])
                    op = f'common_state_gap_v1/results/open/{target}/{model}/seed_{seed}/{s["state_id"]}/summary.json'
                    summary = raw.json(op)
                    all_refs.append(op)
                    require((summary['target'], summary['model'], summary['seed'], summary['state_id']) ==
                            (target, model, seed, s['state_id']), 'open state identity mismatch: ' + op)
                    require(raw.reference(summary['state_path']) == state, 'saved state mismatch: ' + op)
                    b = summary['branches'][0]
                    require(b['kind'] == 'mean_plan', 'plan selector mismatch: ' + op)
                    ar = op.replace('summary.json', 'mean_plan.npz')
                    raw.register(ar, 'state-level predicted and observed reward-head arrays')
                    all_refs.append(ar)
                    with np.load(raw.path(ar), allow_pickle=False) as z:
                        require(np.array_equal(z['actions'], z['plan'][:len(z['actions'])]), 'executed plan mismatch: ' + ar)
                        require(len(z['actions']) == b['steps'], 'branch length mismatch: ' + ar)
                        require(bool(b['complete25']) == (not b['fell'] and b['steps'] == 25), 'completion mismatch: ' + op)
                        if not b['complete25']:
                            require(b['D'] is None, 'noncomplete branch has D: ' + op)
                            continue
                        for field in ['predicted_rewards', 'actual_rewards_head']:
                            require(z[field].shape == (25,) and np.isfinite(z[field]).all(), 'invalid D array: ' + ar)
                        weights = .99**np.arange(25)
                        d = float(np.sum(z['predicted_rewards'] * weights) - np.sum(z['actual_rewards_head'] * weights))
                        require(abs(d - finite(b['D'], 'stored D')) < 3e-6, 'raw/stored D mismatch: ' + ar)
                        values.append(d)
            require(shared is None or selected == shared, 'models selected different common states: ' + target)
            shared = selected
            require(values, 'no valid common-state D: ' + target + '/' + model)
            model_stats[model] = {'mean': statistics.mean(values), 'n_eval': len(values), 'n_selected': len(selected),
                                  'checkpoint':raw.bindings[target,model]}
        adapted = [model_stats[m]['mean'] for m in models if m != 'm0']
        output.append({'target': target, 'm0_mean': model_stats['m0']['mean'], 'adapted_min': min(adapted),
                       'adapted_max': max(adapted), 'n_eval': model_stats['m0']['n_eval'],
                       'n_selected': len(shared), 'n_seed': 20,
                       'train_command_mps': 1., 'eval_command_mps': .75 if target == 'flat' else 1.,
                       'model_coverage': json.dumps(model_stats, sort_keys=True, separators=(',', ':')),
                       'basis': 'identical selected states; checkpoint-specific mean_plan; state mean conditional on model-specific nonterminated25 completion',
                       'source': f'common_state_gap_v1/results/open/{target}',
                       'source_sha256_set': source_set_hash(raw, all_refs)})
    return output


def calculate(raw):
    tables, problems = {}, []
    manifest = raw.json('common_state_gap_v1/manifest.json', 'checkpoint/condition binding only; aggregate numbers ignored')
    for entry in manifest:
        target, model = entry['target'], entry['model']
        raw.bindings[target, model] = raw.checkpoint(entry['checkpoint'], 'evaluation:' + target + ':' + model)
    flat_manifest = raw.json('flat_velocity_matched_v1/manifest.json', 'checkpoint/dataset/condition bindings only')
    require(flat_manifest['adaptation_velocity'] == .75 and flat_manifest['common_seeds'] == sorted(HELDOUT), 'matched flat protocol differs')
    for model, checkpoint in flat_manifest['models'].items():
        raw.bindings['flat', model] = raw.checkpoint(checkpoint, 'evaluation:flat:' + model)
    for producer in ['eval_s1_v2.py', 'module_combo_v1/gap_config.py',
                     'module_combo_v1/evaluate_latent_datasets.py',
                     'positive_control_variance_v1/run.py', 'multi_gap_200seed_extension_v1/settings.py',
                     'gap_diagnosis_v1/analyze.py', 'common_state_gap_v1/worker.py',
                     'tile_failure_data_reanalysis_v1/analyze.py', 'flat_velocity_matched_v1/analyze.py']:
        raw.register(producer, 'producer/method provenance only; code never executed')
    episodes = {}
    for target, models in MODELS.items():
        for model in models:
            episodes[target, model] = load_episodes(raw, target, model)
    for model in ['flat_chain1', 'flat_chain2']:
        episodes['flat', model] = load_episodes(raw, 'flat', model)
    mass_loss = loss_csv(raw, 'positive_control_variance_v1/results/mass8/losses.csv',
                         'independent_m0', MASS, set(range(49301, 49311)))
    tables[FILES[0]] = []
    for model in MASS:
        ev = episodes['mass8', model]
        row = count_row(raw, 'mass8', model, ev, 'full')
        row['failures_100_old'] = sum(ev[s]['failure'] for s in OLD)
        row.update(mass_loss[model])
        tables[FILES[0]].append(row)
    # A status row has no inferred p/statistic; outcomes alone do not identify a test.
    tables[FILES[1]] = [{'test': 'same 200-seed failure proportion across six adapted mass chains',
                        'scope': 'four P0-collected and two policy-modified-collected chains',
                        'approx_p_value': '', 'evidence_status': 'NOT_REPRODUCED',
                        'source_report_section': 'original method/statistic/exact p unavailable',
                        'n_eval': 200, 'n_training_chains': 6,
                        'paired_seed_ranges': '23501-23600;24001-24100',
                        'failure_counts_by_chain': json.dumps({m: sum(x['failure'] for x in episodes['mass8',m].values()) for m in MASS if m!='m0'},sort_keys=True),
                        'basis': 'shared evaluation seeds across six distinct adaptation chains; not 1200 independent observations',
                        'source_sha256_set': source_set_hash(raw,[x['source'] for m in MASS if m!='m0' for x in episodes['mass8',m].values()])}]
    tables[FILES[2]] = value_mc(raw)
    tables[FILES[3]] = common_dynamics(raw)
    tables[FILES[4]] = []
    for target, models in MODELS.items():
        losses = (loss_csv(raw, 'flat_velocity_matched_v1/results/losses.csv', 'common075', models, HELDOUT)
                  if target == 'flat' else common_losses(raw, target, models))
        baseline = losses['m0']['loss25']
        require(baseline > 0, 'zero baseline loss: ' + target)
        for model in models:
            row = count_row(raw, target, model, episodes[target, model], 'full' if target == 'flat' else 'tile')
            row.update(losses[model])
            row['loss_reduction_pct'] = 100 * (1 - row['loss25'] / baseline)
            row['train_command_mps'] = .75 if model.startswith('flat075') else ('' if model == 'm0' else 1.)
            tables[FILES[4]].append(row)
    tables[FILES[5]] = []
    for model in ['m0', 'flat_chain1', 'flat_chain2', 'flat075_chain1', 'flat075_chain2']:
        row = count_row(raw, 'flat', model, episodes['flat', model], 'full')
        row['train_command_mps'] = '' if model == 'm0' else (.75 if model.startswith('flat075') else 1.)
        tables[FILES[5]].append(row)
    return tables, problems


def verify_expectations(tables):
    """README/request values are expectations only, never calculation inputs."""
    differences = []
    mass = {r['model']: r for r in tables[FILES[0]]}
    expectations = {'m0': (.0642, 62, 64.5), 'old_chain1': (.0442, 40, 43.5),
                    'old_chain2': (.0375, 49, 48.), 'new_chain3': (.0352, 37, 40.),
                    'new_chain4': (.0379, 52, 58.5)}
    for model, expected in expectations.items():
        for field, value in zip(['loss25', 'failures_100_old', 'failure_pct'], expected):
            observed = mass[model][field]
            tolerance = .00005 if field == 'loss25' else 1e-10
            differences.append({'table': FILES[0], 'model': model, 'field': field,
                                'expected': value, 'actual': observed, 'delta': observed-value,
                                'pass': abs(observed-value) <= tolerance})
    for row, expected in zip(tables[FILES[2]], [(.57, .76), (.57, .77), (.39, .68), (.05, .44)]):
        for field, value in zip(['rho_head_mc', 'rho_critic_mc'], expected):
            differences.append({'table': FILES[2], 'target': row['target'], 'field': field,
                                'expected': value, 'actual': row[field], 'delta': row[field]-value,
                                'pass': abs(row[field]-value) <= .005})
    for row, expected in zip(tables[FILES[3]], [(.152, .100, .111), (.308, .255, .271), (.171, .144, .146), (.345, .230, .256)]):
        for field, value in zip(['m0_mean', 'adapted_min', 'adapted_max'], expected):
            differences.append({'table': FILES[3], 'target': row['target'], 'field': field,
                                'expected': value, 'actual': row[field], 'delta': row[field]-value,
                                'pass': abs(row[field]-value) <= .0005})
    # The README lists sorted, unnamed adaptation rates. Never assign that order
    # to named models; retain discovered model identities in the detailed rows.
    failures = {'mass8': [62.5, 38., 41.5, 43.5, 46.5, 57., 58.],
                'friction015': [81.5, 74., 74.5, 79., 87.5],
                'gain070': [58., 43., 64.5], 'flat': [53., 58., 62.]}
    for target, wanted in failures.items():
        rr = [r for r in tables[FILES[4]] if r['target'] == target]
        selected = [next(r for r in rr if r['model'] == 'm0')] + sorted(
            [r for r in rr if r['model'] != 'm0'], key=lambda r: r['failure_pct'])
        for row, value in zip(selected, wanted):
            differences.append({'table': FILES[4], 'target': target, 'model': row['model'], 'field': 'failure_pct',
                                'expected': value, 'actual': row['failure_pct'], 'delta': row['failure_pct']-value,
                                'pass': row['failure_pct'] == value})
    for row, value in zip(tables[FILES[5]], [53., 91.5, 94.5, 62., 58.]):
        differences.append({'table': FILES[5], 'model': row['model'], 'field': 'failure_pct', 'expected': value,
                            'actual': row['failure_pct'], 'delta': row['failure_pct']-value, 'pass': row['failure_pct'] == value})
    for target, bounds in {'mass8': (32, 46), 'friction015': (42, 46),
                           'gain070': (43, 45), 'flat': (84, 85)}.items():
        values = [r['loss_reduction_pct'] for r in tables[FILES[4]] if r['target'] == target and r['model'] != 'm0']
        for field, value, wanted in [('reduction_min_pct', min(values), bounds[0]),
                                     ('reduction_max_pct', max(values), bounds[1])]:
            differences.append({'table': FILES[4], 'target': target, 'field': field,
                                'actual': value, 'expected_rounded': wanted, 'pass': round(value) == wanted})
    return differences


def csv_text(rows):
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


TARGET_LABELS = {'mass8': 'mass +8kg', 'friction015': 'friction 0.15',
                 'gain070': 'gain 0.7x', 'flat': 'flat friction 0.10', 'source': 'source'}


def compact_tables(details):
    """Preserve the discovered six CSV headers and row cardinalities.

    Detailed precision, denominators, conditions, sources and hashes reside in
    the generated PROVENANCE.json, rather than silently changing compact schemas.
    """
    result = {}
    labels = {'m0': 'M0', 'old_chain1': 'Chain 1 / old_chain1', 'old_chain2': 'Chain 2 / old_chain2',
              'new_chain3': 'Chain 3 / new_chain3', 'new_chain4': 'Chain 4 / new_chain4',
              'old_DP': 'old_DP', 'new_DP2': 'new_DP2'}
    result[FILES[0]] = [
        {'chain': labels[r['model']], 'collection_policy': 'pretrained' if r['model']=='m0' else
         ('policy-adapted data collection' if r['model'] in ('old_DP','new_DP2') else 'P0'),
         'latent_mse_25': round(r['loss25'],4), 'failures_100': r['failures_100_old'],
         'failures_200_pct': r['failure_pct'], 'basis': 'whole episode; up to 1000 steps',
         'source_report_section': 'raw-derived; PROVENANCE.json'} for r in details[FILES[0]]]
    r = details[FILES[1]][0]
    result[FILES[1]] = [{key:r[key] for key in ['test','scope','approx_p_value','evidence_status','source_report_section']}]
    result[FILES[2]] = [
        {'environment': TARGET_LABELS[r['target']], 'value_head_spearman_mc': round(r['rho_head_mc'],2),
         'privileged_critic_spearman_mc': round(r['rho_critic_mc'],2),
         'evaluation_basis': f"expert continuation after step25; MC horizon {r['mc_horizon']}; discount0.99; pooled snapshots",
         'source_report_section': 'raw-derived; PROVENANCE.json'} for r in details[FILES[2]]]
    result[FILES[3]] = [
        {'target': TARGET_LABELS[r['target']], 'm0_reward_d_error': round(r['m0_mean'],3),
         'adapted_reward_d_min': round(r['adapted_min'],3), 'adapted_reward_d_max': round(r['adapted_max'],3),
         'adaptation_train_speed_mps': r['train_command_mps'], 'evaluation_speed_mps': r['eval_command_mps'],
         'basis': 'matched target-specific states; checkpoint mean_plan; model-specific complete25 mask',
         'source_report_section': 'raw-derived; PROVENANCE.json'} for r in details[FILES[3]]]
    result[FILES[4]] = []
    for target in MODELS:
        rows = [r for r in details[FILES[4]] if r['target']==target]
        base = next(r for r in rows if r['model']=='m0')
        adapted = [r for r in rows if r['model']!='m0']
        percentages = sorted(r['failure_pct'] for r in adapted)
        result[FILES[4]].append(
            {'target': TARGET_LABELS[target], 'failure_basis': 'entire 1000-step episode' if target=='flat' else 'inside boxes tile',
             'speed_mps': base['eval_command_mps'], 'm0_failure_pct': base['failure_pct'],
             'adapted_failure_pct_list': ';'.join(f'{x:g}' for x in percentages),
             'loss_reduction_pct_approx': f"{round(min(r['loss_reduction_pct'] for r in adapted))}-{round(max(r['loss_reduction_pct'] for r in adapted))}",
             'evaluation_n': 200, 'source_report_section': 'raw-derived target held-out; PROVENANCE.json'})
    base = next(r for r in details[FILES[5]] if r['model']=='m0')
    result[FILES[5]] = [
        {'adaptation_speed_mps': velocity, 'evaluation_speed_mps': .75, 'm0_failure_pct': base['failure_pct'],
         'adapted_failure_pct_list': ';'.join(f'{x:g}' for x in sorted(r['failure_pct'] for r in details[FILES[5]] if r['train_command_mps']==velocity)),
         'n_eval':200, 'basis':'flat friction 0.10, 1000-step episode',
         'source_report_section':'raw-derived; PROVENANCE.json'} for velocity in (1.,.75)]
    return result


def markdown_tables(text):
    groups, rows = [], []
    for line in text.splitlines() + ['']:
        if line.startswith('|'):
            cells = [x.strip() for x in line.strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',x) for x in cells):
                rows.append(cells)
        elif rows:
            groups.append(rows)
            rows=[]
    return groups


def validate_readme(details):
    """Parse actual README cells; do not use a whole-file pin for numeric claims."""
    text = (REPO/'README.md').read_text()
    groups = markdown_tables(text)
    comparisons=[]
    def check(label, text_value, calculated, tolerance):
        value=finite(text_value.rstrip('%'), 'README '+label)
        comparisons.append({'claim':label,'readme':value,'raw':calculated,
                            'pass':abs(value-calculated)<=tolerance+1e-12})
    mass=next((g for g in groups if g[0]==['Chain','25-step latent MSE','실패 / 100','실패율 / 200']),None)
    require(mass is not None and len(mass)==6, 'README mass table missing/changed schema')
    for cells,r in zip(mass[1:],details[FILES[0]][:5]):
        require(cells[0]==('M0' if r['model']=='m0' else 'Chain '+str(MASS.index(r['model']))),'README chain name mismatch')
        for label,cell,number,tolerance in [('loss25',cells[1],r['loss25'],.00005),('failures100',cells[2],r['failures_100_old'],0),('failure_pct',cells[3],r['failure_pct'],0)]:
            check(r['model']+':'+label,cell,number,tolerance)
    value=next((g for g in groups if g[0]==['환경','Value head 대 MC','Privileged critic 대 MC']),None)
    require(value is not None and len(value)==5,'README value table missing/changed schema')
    for cells,r in zip(value[1:],details[FILES[2]]):
        require(cells[0]=={'source':'Source','mass8':'Mass +8 kg','friction015':'Friction 0.15','gain070':'Gain 0.7배'}[r['target']], 'README value target mismatch')
        for field,cell in [('rho_head_mc',cells[1]),('rho_critic_mc',cells[2])]:check(r['target']+':'+field,cell,r[field],.005)
    dynamics=next((g for g in groups if g[0]==['Target','M0 오차','Adapted 오차']),None)
    require(dynamics is not None and len(dynamics)==5,'README dynamics table missing/changed schema')
    for cells,r in zip(dynamics[1:],details[FILES[3]]):
        require(cells[0]=={'mass8':'Mass +8 kg','friction015':'Friction 0.15','gain070':'Gain 0.7배','flat':'평지 friction 0.10'}[r['target']], 'README D target mismatch')
        parts=cells[2].split('~');require(len(parts)==2,'README D interval invalid')
        for field,cell in [('m0_mean',cells[1]),('adapted_min',parts[0]),('adapted_max',parts[1])]:check(r['target']+':'+field,cell,r[field],.0005)
    failures=next((g for g in groups if g[0]==['Target','25-step loss 감소','M0 실패율','Dynamics 적응 후','해석']),None)
    require(failures is not None and len(failures)==5,'README target table missing/changed schema')
    for cells,target in zip(failures[1:],MODELS):
        require(cells[0]=={'mass8':'Mass +8 kg','friction015':'Friction 0.15','gain070':'Gain 0.7배','flat':'평지 friction 0.10 (0.75 m/s 재적응)'}[target], 'README failure target mismatch')
        rows=[r for r in details[FILES[4]] if r['target']==target];base=next(r for r in rows if r['model']=='m0');adapted=[r for r in rows if r['model']!='m0']
        check(target+':baseline_failure',cells[2],base['failure_pct'],0)
        stated=sorted(float(x) for x in re.findall(r'\d+(?:\.\d+)?',cells[3]))
        require(stated==sorted(r['failure_pct'] for r in adapted),'README adapted failure list mismatch: '+target)
        interval=re.search(r'(\d+)~(\d+)%',cells[1]);require(interval is not None,'README loss interval missing')
        require([int(x) for x in interval.groups()]==[round(min(r['loss_reduction_pct'] for r in adapted)),round(max(r['loss_reduction_pct'] for r in adapted))], 'README held-out family/range mismatch: '+target)
    require('MC horizon은 25-step 뒤 continuation 기준 source/mass에서 175 step, friction/gain에서 200 step' in text,
            'README must map historical MC horizons to their correct targets')
    # Exact held-out reductions, seed coverage and denominators are validated
    # from raw inputs and in generated provenance; README displays rounded cells.
    mass_loss={r['model']:r['loss25'] for r in details[FILES[0]]}
    independent=[100*(1-mass_loss[m]/mass_loss['m0']) for m in MASS[1:5]]
    independent_cell=re.search(r'4절 검증 세트:\s*(\d+)~(\d+)%',failures[1][1])
    require(independent_cell is not None and
            [int(x) for x in independent_cell.groups()]==[round(min(independent)),round(max(independent))]==[31,45],
            'README independent loss family interval differs')
    require(not re.search(r'p\s*[≈=]\s*0\.0002|기각됐',text),'README still asserts unreproduced six-chain test')
    require(all(x['pass'] for x in comparisons),'README numeric mismatch')
    return comparisons


def public_provenance(raw, details, comparisons):
    families={}
    public_inputs=[]
    for relative,record in raw.inputs.items():
        families.setdefault(relative.split('/')[0],[]).append(relative)
        if relative.endswith('.py') or relative in ['positive_control_variance_v1/results/mass8/losses.csv',
                'flat_velocity_matched_v1/results/losses.csv','common_state_gap_v1/manifest.json','flat_velocity_matched_v1/manifest.json']:
            public_inputs.append(record)
    checkpoints=[]
    for binding in raw.checkpoints.values():
        checkpoints.append({key:binding[key] for key in ['ids','step','sha256_tree']})
    return {'schema_version':2,'measurement_status':'REPRODUCED','original_six_chain_test_status':'NOT_REPRODUCED',
            'original_p_value':None,'substitute_test_performed':False,
            'builder_sha256':digest(Path(__file__)),
            'detail_rows':details,'readme_comparisons':comparisons,
            'input_file_count':len(raw.inputs),'direct_sources':sorted(public_inputs,key=lambda x:x['path']),
            'input_families':[{'family':family,'files':len(refs),'sha256_set':source_set_hash(raw,refs)} for family,refs in sorted(families.items())],
            'checkpoints':sorted(checkpoints,key=lambda x:x['ids'][0]),
            'full_file_manifest':'LOCAL_ONLY: analysis/.local_audit/',
            'hash_method':'SHA256 of sorted UTF8 relative_path TAB file_SHA256 LF; checkpoint tree uses checkpoint-relative paths',
            'public_scope':'aggregate rows, explicit producer/per-seed loss source names, family-set digests and logical checkpoint IDs; no per-episode/state filename inventory',
            'aggregation':'independent seed means for loss; pooled snapshots for Spearman; conditional state means for D; paired evaluation seeds across training chains',
            'mc_cross_target_comparability':'different horizons175/200; no direct cross-target MC return comparison',
            'source_integrity':'all registered raw inputs and checkpoint files checked unchanged after read; no source writes'}


def compare_csv(path, expected):
    if not path.is_file():
        return 'missing existing CSV'
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        actual = list(reader)
    if header != list(expected[0]):
        return compare_historical_csv(path.name, header, actual, expected)
    if len(actual) != len(expected):
        return 'row-count mismatch'
    # Row order/naming, conditions, n, hashes and method strings must also match.
    for index, (a, e) in enumerate(zip(actual, expected)):
        for field, value in e.items():
            if isinstance(value, (int, float)):
                try:
                    observed = finite(a[field], field)
                except (ValueError, TypeError):
                    return f'invalid numeric cell row {index + 1}: {field}'
                tolerance = 0 if isinstance(value, int) else 1e-10
                if abs(observed-value) > tolerance:
                    return f'numeric mismatch row {index + 1}: {field}'
            elif a[field] != str(value):
                return f'identity/method/source mismatch row {index + 1}: {field}'
    return None


def compare_historical_csv(name, header, actual, expected):
    """Explicit mappings of discovered compact CSVs, for verification only.

    Legacy report references do not become raw input provenance. Approximate
    cells use only their displayed precision, while conditions/n/names are exact.
    """
    schemas = {
        FILES[0]: ['chain', 'collection_policy', 'latent_mse_25', 'failures_100', 'failures_200_pct', 'basis', 'source_report_section'],
        FILES[2]: ['environment', 'value_head_spearman_mc', 'privileged_critic_spearman_mc', 'evaluation_basis', 'source_report_section'],
        FILES[3]: ['target', 'm0_reward_d_error', 'adapted_reward_d_min', 'adapted_reward_d_max', 'adaptation_train_speed_mps', 'evaluation_speed_mps', 'basis', 'source_report_section'],
        FILES[4]: ['target', 'failure_basis', 'speed_mps', 'm0_failure_pct', 'adapted_failure_pct_list', 'loss_reduction_pct_approx', 'evaluation_n', 'source_report_section'],
        FILES[5]: ['adaptation_speed_mps', 'evaluation_speed_mps', 'm0_failure_pct', 'adapted_failure_pct_list', 'n_eval', 'basis', 'source_report_section'],
    }
    if header != schemas.get(name):
        return 'unrecognized historical CSV schema'
    aliases = {'source': 'source', 'mass +8kg': 'mass8', 'friction 0.15': 'friction015',
               'gain 0.7x': 'gain070', 'flat friction 0.10': 'flat'}
    seen = set()
    issues = []

    def number(row, field, value, tolerance=0.):
        observed = finite(row[field], 'comparison ' + field)
        if abs(observed-value) > tolerance + 1e-12:
            issues.append(f'{field}: existing={observed:.12g}, raw={value:.12g}')

    def rates(text):
        return sorted(finite(x, 'failure percentage') for x in text.split(';'))

    if name == FILES[0]:
        names = {'M0': 'm0', 'Chain 1 / old_chain1': 'old_chain1', 'Chain 2 / old_chain2': 'old_chain2',
                 'Chain 3 / new_chain3': 'new_chain3', 'Chain 4 / new_chain4': 'new_chain4',
                 'old_DP': 'old_DP', 'new_DP2': 'new_DP2'}
        for row in actual:
            require(row['chain'] in names, 'unexpected mass chain comparison name')
            model = names[row['chain']]
            require(model not in seen, 'duplicate historical comparison row')
            seen.add(model)
            r = next(x for x in expected if x['model'] == model)
            number(row, 'latent_mse_25', r['loss25'], .00005)
            number(row, 'failures_100', r['failures_100_old'])
            number(row, 'failures_200_pct', r['failure_pct'])
            policy = 'pretrained' if model == 'm0' else ('policy-adapted data collection' if model in ['old_DP', 'new_DP2'] else 'P0')
            require(row['collection_policy'] == policy and row['basis'] == 'whole episode; up to 1000 steps',
                    'mass comparison condition mismatch')
        require(seen == set(MASS), 'historical mass comparison missing row')
    elif name in (FILES[2], FILES[3]):
        key = 'environment' if name == FILES[2] else 'target'
        for row in actual:
            require(row[key] in aliases, 'unexpected target comparison name')
            target = aliases[row[key]]
            require(target not in seen, 'duplicate historical target')
            seen.add(target)
            r = next(x for x in expected if x['target'] == target)
            fields = ([('value_head_spearman_mc', 'rho_head_mc'), ('privileged_critic_spearman_mc', 'rho_critic_mc')]
                      if name == FILES[2] else [('m0_reward_d_error', 'm0_mean'), ('adapted_reward_d_min', 'adapted_min'), ('adapted_reward_d_max', 'adapted_max')])
            for old, new in fields:
                number(row, old, r[new], .005 if name == FILES[2] else .0005)
            if name == FILES[3]:
                number(row, 'adaptation_train_speed_mps', r['train_command_mps'])
                number(row, 'evaluation_speed_mps', r['eval_command_mps'])
        require(seen == {r['target'] for r in expected}, 'historical comparison target coverage mismatch')
    elif name == FILES[4]:
        for row in actual:
            require(row['target'] in aliases, 'unexpected failure-summary target')
            target = aliases[row['target']]
            require(target not in seen, 'duplicate failure-summary target')
            seen.add(target)
            rr = [r for r in expected if r['target'] == target]
            m0 = next(r for r in rr if r['model'] == 'm0')
            adapted = [r for r in rr if r['model'] != 'm0']
            number(row, 'm0_failure_pct', m0['failure_pct'])
            number(row, 'speed_mps', m0['eval_command_mps'])
            number(row, 'evaluation_n', 200)
            require(row['failure_basis'] == ('entire 1000-step episode' if target == 'flat' else 'inside boxes tile'),
                    'failure-summary basis mismatch')
            if rates(row['adapted_failure_pct_list']) != sorted(r['failure_pct'] for r in adapted):
                issues.append('adapted failure-rate list differs: ' + target)
            interval = [finite(x, 'loss reduction interval') for x in row['loss_reduction_pct_approx'].split('-')]
            require(len(interval) == 2, 'invalid historical reduction interval')
            recomputed = [round(min(r['loss_reduction_pct'] for r in adapted)), round(max(r['loss_reduction_pct'] for r in adapted))]
            if interval != recomputed:
                issues.append(f'{target} loss-reduction interval: existing={interval}, raw-rounded={recomputed}')
        require(seen == set(MODELS), 'failure-summary target coverage mismatch')
    elif name == FILES[5]:
        m0 = next(r for r in expected if r['model'] == 'm0')
        for row in actual:
            speed = finite(row['adaptation_speed_mps'], 'adaptation speed')
            require(speed in (1., .75) and speed not in seen, 'unknown/duplicate adaptation condition')
            seen.add(speed)
            number(row, 'evaluation_speed_mps', .75)
            number(row, 'm0_failure_pct', m0['failure_pct'])
            number(row, 'n_eval', 200)
            computed = sorted(r['failure_pct'] for r in expected if r['train_command_mps'] == speed)
            if rates(row['adapted_failure_pct_list']) != computed:
                issues.append('adapted velocity-matched rates differ')
        require(seen == {1., .75}, 'missing adaptation condition')
    return '; '.join(issues) if issues else None


def evidence_readme():
    """Single publication template; checked independently of README prose."""
    return (
        '# Raw-derived portfolio evidence\n\n'
        'Five measurement tables are regenerated from saved seed/episode/state measurements. '
        '`chain_equality_test.csv` is a NOT_REPRODUCED status record with an empty p-value. '
        'No replacement statistical test was performed.\n\n'
        'The six historical CSV headers and row counts are preserved. '
        '`source_report_section` now points to raw-derived provenance, rather than report transcription.\n\n'
        'Section 4 loss: independent seeds 49301–49310, arithmetic seed mean. '
        'Section 7 loss: target common held-out seeds 186001–186020, arithmetic seed mean; '
        'mass reduction 32.299027–46.356004%. Control evaluations use '
        '23501–23600 plus 24001–24100.\n\n'
        'MC continuation is 175 steps for source/mass and 200 for friction/gain. '
        'Head and critic share identical within-target pairs; cross-target MC returns are not directly compared.\n\n'
        'See [PROVENANCE.json](PROVENANCE.json) for exact values, per-model counts, checkpoint IDs, '
        'source selections, SHA256 and per-file publication status; '
        '[method documentation](../../analysis/PORTFOLIO_RAW_PROVENANCE.md) for schemas and controls. '
        'The full per-file input inventory is local-only and excluded from Git.\n')


def publish(out, tables, audit):
    """Publish one complete generation by atomic rename/exchange on Linux.

    All failure gates run before staging. An existing directory uses renameat2
    RENAME_EXCHANGE so readers never observe a missing or mixed generation.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.parent / ('.' + out.name + '.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage = Path(tempfile.mkdtemp(prefix='.' + out.name + '.', dir=out.parent))
    try:
        for name in FILES:
            if name in tables:
                (stage / name).write_text(csv_text(tables[name]))
            elif (out / name).is_file():
                shutil.copyfile(out / name,stage / name)
        (stage / 'PROVENANCE.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        (stage / 'README.md').write_text(evidence_readme())
        if out.exists():
            require(out.is_dir() and not out.is_symlink(), 'unsafe output directory')
            allowed = set(FILES) | {'PROVENANCE.json', 'README.md'}
            require({p.name for p in out.iterdir()} <= allowed, 'output contains unrelated files')
            libc = ctypes.CDLL(None, use_errno=True)
            require(hasattr(libc, 'renameat2'), 'atomic directory exchange unavailable')
            exchange = libc.renameat2
            exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            exchange.restype = ctypes.c_int
            result = exchange(-100, os.fsencode(stage), -100, os.fsencode(out), 2)
            require(result == 0, 'atomic directory exchange failed; existing generation preserved')
        else:
            stage.rename(out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        os.close(fd)
        lock.unlink()


def publication_gate(out, name, compact, detail):
    p=out/name
    if not p.exists():return None
    if (out/'PROVENANCE.json').exists():
        return compare_csv(p,compact)
    if name==FILES[1]:
        with p.open(newline='') as f:
            reader=csv.DictReader(f);rows=list(reader)
        require(reader.fieldnames==list(compact[0]) and len(rows)==1,'unrecognized historical test schema')
        row=rows[0]
        require(row['test']==compact[0]['test'] and row['scope']==compact[0]['scope'],'historical test scope differs')
        require(finite(row['approx_p_value'],'reported approximate p')==.0002,'historical test expectation changed')
        return None  # Explicit migration to empty p and NOT_REPRODUCED, not a recomputation.
    with p.open(newline='') as f:
        reader=csv.DictReader(f);rows=list(reader);header=reader.fieldnames
    reason=compare_historical_csv(name,header,rows,detail)
    if name==FILES[4] and reason=='mass8 loss-reduction interval: existing=[31.0, 45.0], raw-rounded=[32, 46]':
        return None  # Authorized correction of the demonstrated validation-family mix-up.
    return reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', default=os.environ.get('SIMDIST_RAW_ROOT'))
    parser.add_argument('--out', default='results/portfolio_evidence_20261009')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if not args.raw_root or not Path(args.raw_root).is_dir():
        print(json.dumps({'status': 'FAIL', 'reason': 'Set SIMDIST_RAW_ROOT or --raw-root to an existing scratch directory.'}))
        return 2
    out = (REPO / args.out).resolve()
    if not out.is_relative_to(REPO) or out == REPO:
        print(json.dumps({'status': 'FAIL', 'reason': 'Output must be a child directory of the portfolio repository.'}))
        return 2
    raw = Raw(Path(args.raw_root))
    details, problems = {}, []
    try:
        details, problems = calculate(raw)
        regressions=verify_expectations(details)
        require(all(x['pass'] for x in regressions),'historical numeric expectation mismatch')
        comparisons=validate_readme(details)
        tables=compact_tables(details)
        gates={}
        selected={}
        frozen={name:digest(out/name) for name in FILES if (out/name).is_file()}
        for name in FILES:
            try:
                reason=publication_gate(out,name,tables[name],details[name])
            except (EvidenceError, ValueError, KeyError, OSError) as exc:
                reason=str(exc) if isinstance(exc,EvidenceError) else 'existing CSV schema/read failure: '+type(exc).__name__
            if reason:
                gates[name]={'status':'SKIP','reason':reason}
                problems.append({'table':name,'reason':reason})
            else:
                gates[name]={'status':'VALIDATED_STATUS_ONLY' if name==FILES[1] else 'VALIDATED_RAW_MEASUREMENT',
                             'rows':len(tables[name])}
                selected[name]=tables[name]
            if args.check:
                try:
                    mismatch=compare_csv(out/name,tables[name])
                except (ValueError, KeyError, OSError):
                    mismatch='existing CSV schema/read failure'
                if mismatch and not reason:
                    problems.append({'table':name,'reason':'output requires generation: '+mismatch})
        raw.unchanged()
        public=public_provenance(raw,details,comparisons)
        public['per_file_validation']=gates
        public['regression_comparisons']=regressions
        if args.write:
            require(all(digest(out/name)==sha for name,sha in frozen.items()),'output changed since validation')
            local=REPO/'analysis/.local_audit'
            local.mkdir(exist_ok=True)
            full={'inputs':sorted(raw.inputs.values(),key=lambda x:x['path']),
                  'checkpoints':raw.checkpoints,'input_stats_unchanged':True,
                  'regression_comparisons':regressions,'details':details,'per_file_validation':gates}
            (local/'full_input_manifest.json').write_text(json.dumps(full,indent=2,allow_nan=False)+'\n')
            if selected:publish(out,selected,public)
        else:
            if not (out/'PROVENANCE.json').is_file():
                problems.append({'reason':'missing generated provenance'})
            elif json.loads((out/'PROVENANCE.json').read_text())!=public:
                problems.append({'reason':'generated provenance differs from recomputed inputs/methods'})
            if not (out/'README.md').is_file() or (out/'README.md').read_text()!=evidence_readme():
                problems.append({'reason':'generated evidence README differs from publication template'})
    except (EvidenceError, KeyError, ValueError, OSError) as exc:
        # Do not echo arbitrary exceptions: parsers/OSError can contain private paths.
        reason = str(exc) if isinstance(exc, EvidenceError) else 'Input/schema/runtime failure: ' + type(exc).__name__
        problems.append({'reason':reason})
    report={'status':'FAIL' if problems else 'VALIDATED_WITH_NOT_REPRODUCED_TEST',
            'original_test_status':'NOT_REPRODUCED','original_p_value':None,
            'measurement_rows':sum(len(v) for k,v in details.items() if k!=FILES[1]),
            'issues':problems,'input_files_hashed':len(raw.inputs),
            'raw_modified':False,'experiments_started':False,'mode':'write' if args.write else 'check'}
    print(json.dumps(report,indent=2,allow_nan=False))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
