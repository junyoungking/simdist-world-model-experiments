"""Combine a dynamics checkpoint with policy heads from another checkpoint."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import argparse
import json
import shutil
from pathlib import Path
import flax.nnx as nnx
import orbax.checkpoint as ocp
import yaml
from simdist.utils import model as model_utils
parser = argparse.ArgumentParser()
parser.add_argument('--dynamics', type=Path, required=True)
parser.add_argument('--policy', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
dynamics, dynamics_cfg, step = model_utils.load_model_from_ckpt(str(args.dynamics.resolve()), None)
policy, _, policy_step = model_utils.load_model_from_ckpt(str(args.policy.resolve()), None)
policy_filter = model_utils.create_param_filter(['policy', 'policy_dec'])
policy_state = nnx.state(policy, policy_filter)
nnx.update(dynamics, policy_state)
checkpoint = args.out / 'checkpoint'
checkpoint.mkdir()
shutil.copy2(args.dynamics / 'model_config.yaml', checkpoint / 'model_config.yaml')
with ocp.CheckpointManager(str(checkpoint), options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True, enable_async_checkpointing=False)) as manager:
    manager.save(step, args=ocp.args.StandardSave(nnx.state(dynamics).to_pure_dict()))
summary = {'dynamics_checkpoint': str(args.dynamics.resolve()), 'dynamics_step': step, 'policy_checkpoint': str(args.policy.resolve()), 'policy_step': policy_step, 'transplanted_heads': ['policy', 'policy_dec'], 'checkpoint': str(checkpoint.resolve())}
(args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary), flush=True)
