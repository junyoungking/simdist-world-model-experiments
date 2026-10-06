"""Frozen world-model inference on candidate and observed latent sequences."""
import sys as _portfolio_sys
from pathlib import Path as _PortfolioPath
_portfolio_root = next((p for p in _PortfolioPath(__file__).resolve().parents if (p / 'portfolio_paths.py').is_file()))
_portfolio_sys.path.insert(0, str(_portfolio_root))
from portfolio_paths import PORTFOLIO_ROOT, artifact_root, artifact_dir, legacy_dir, document_path, simdist_root, checkpoint_root, resolve_artifact_path, configure_imports, result_path
configure_imports()
import jax
import jax.numpy as jnp
from flax import nnx
from simdist.utils.model import repeat_along_batch_dim

@nnx.jit
def score_actions(model, model_inputs, actions, discounts, final_discount):
    x = repeat_along_batch_dim(model_inputs, actions.shape[0])
    x['fut_acts'] = actions
    y = model.inference(x)
    reward_sum = jnp.sum(y['rewards'] * discounts[None, :], axis=-1)
    value_raw = y['values'][:, -1]
    value_discounted = value_raw * final_discount
    return (y['rewards'], value_raw, reward_sum, value_discounted, reward_sum + value_discounted)

@nnx.jit
def observed_heads(model, proprio, extero, actions, commands):
    """Observed z_t..z_t+25; reward input z_t..z_t+24; value z_t+1..z_t+25."""
    scaler = model.get_scaler()
    scaled_obs = scaler.scale({'proprio_obs': proprio, 'extero_obs': extero})
    latents = model.encode_latent(scaled_obs['proprio_obs'], scaled_obs['extero_obs'], deterministic=True)
    scaled_inputs = scaler.scale({'fut_acts': actions, 'fut_cmds': commands})
    reward_in = jnp.concatenate((latents[:, :-1], scaled_inputs['fut_acts'], scaled_inputs['fut_cmds'][:, :-1]), axis=-1)
    reward_emb = model.reward_emb(reward_in, deterministic=True)
    reward_seq = model.reward_dec(model.reward(reward_emb, deterministic=True), deterministic=True).squeeze(-1)
    value_in = jnp.concatenate((latents[:, 1:], scaled_inputs['fut_cmds'][:, 1:]), axis=-1)
    value_emb = model.value_emb(value_in, deterministic=True)
    value_seq = model.value_dec(model.value(value_emb, deterministic=True), deterministic=True).squeeze(-1)
    out = scaler.unscale({'rewards': reward_seq, 'values': value_seq})
    return (latents, out['rewards'], out['values'])

@nnx.jit
def state_value_from_observed_history(model, proprio, extero, commands):
    """Value head at the latest real state using the preceding 25 real latents."""
    scaler = model.get_scaler()
    scaled_obs = scaler.scale({'proprio_obs': proprio, 'extero_obs': extero})
    latents = model.encode_latent(scaled_obs['proprio_obs'], scaled_obs['extero_obs'], deterministic=True)
    scaled_command = scaler.scale({'fut_cmds': commands})['fut_cmds']
    value_in = jnp.concatenate((latents, scaled_command), axis=-1)
    value_emb = model.value_emb(value_in, deterministic=True)
    value_seq = model.value_dec(model.value(value_emb, deterministic=True), deterministic=True).squeeze(-1)
    return scaler.unscale({'values': value_seq})['values'][:, -1]
