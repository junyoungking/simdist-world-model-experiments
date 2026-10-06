"""Phase 0 paths, immutable manifests, and released configuration composition."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
from pathlib import Path
import hashlib
import json
import os
REPO = PORTFOLIO_ROOT
ROOT = resolve_artifact_path(os.environ.get('SIMDIST_C_OUTPUT_ROOT', str(artifact_dir('phase_0')))).resolve()
EXPERIMENT_MODE = os.environ.get('SIMDIST_C_EXPERIMENT_MODE')
if EXPERIMENT_MODE not in {'mass', 'friction', 'kp'}:
    raise ValueError('SIMDIST_C_EXPERIMENT_MODE must explicitly be mass, friction, or kp')
MASS_MODE = EXPERIMENT_MODE == 'mass'
if (ROOT / 'preregistered_protocol.json').exists():
    _protocol = json.loads((ROOT / 'preregistered_protocol.json').read_text())
    _registered_mode = _protocol.get('experiment_mode')
    if _registered_mode is None and _protocol.get('protocol_kind') not in {'all_family_carry_forward', 'targetB_friction_mass_fixedgain_reevaluation'}:
        _registered_mode = 'kp' if _protocol.get('phase') in {'1', '1_ratio_map', '1C_retention_control', '1C_retention_control_v2'} else 'mass' if _protocol.get('family') == 'base_added_mass' else 'friction'
    if _registered_mode is not None and EXPERIMENT_MODE != _registered_mode:
        raise ValueError('Explicit experiment mode conflicts with preregistered protocol')
SOURCE = checkpoint_root() / 'models/go2_2026-02-12'

def write_json(path, value):
    path = resolve_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(resolve_artifact_path(directory).rglob('*')) if p.is_file()}

def route_paths():
    from simdist.utils import paths
    data = ROOT / 'datasets'
    data.mkdir(exist_ok=True)
    paths._PATHS['ALL_DATASETS'] = str(data)
    paths._PATHS['REAL_DATASETS'] = str(data)
    paths._PATHS['MODEL_CHECKPOINTS'] = str(ROOT / 'models')

def config(name):
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(simdist_root() / 'config'), version_base=None):
        return OmegaConf.to_container(compose(config_name=name), resolve=True)

def close_dataset(dataset):
    for f in dataset._files.values():
        f.close()
