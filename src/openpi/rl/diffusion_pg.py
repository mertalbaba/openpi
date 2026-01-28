"""Policy-gradient update for diffusion models, similar to REINFORCE style (pi0)."""

from __future__ import annotations

import dataclasses
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as _model
from openpi.training import utils as training_utils


def init_train_state(
    model: _model.BaseModel,
    *,
    tx: optax.GradientTransformation,
    trainable_filter: nnx.filterlib.Filter | None = None,
) -> training_utils.TrainState:
    """Initialize a TrainState for a loaded OpenPi model."""
    if trainable_filter is None:
        trainable_filter = nnx.All(nnx.Param)

    params = nnx.state(model)
    return training_utils.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )


def diffusion_pg_update(
    rng: jax.Array,
    state: training_utils.TrainState,
    policy_observation: _model.Observation,
    critic: Any,
    critic_observation: Any,
    *,
    eta: float = 0.1,
    num_steps: int = 10,
    critic_reduction: str = "min",
    normalize_adv: bool = True,
    adv_clip: float | None = 10.0,
    trainable_filter: nnx.filterlib.Filter | None = None,
) -> tuple[training_utils.TrainState, dict[str, jax.Array]]:
    """Update a diffusion model with log-prob policy gradients.
    `policy_observation` must alredy be in the OpenPi Observation format.
    `critic_observation` should be the input expected by the critic network.
    """
    if trainable_filter is None:
        trainable_filter = nnx.All(nnx.Param)

    model = nnx.merge(state.model_def, state.params)
    model.train()

    @jax.named_call
    def loss_fn(
        model: _model.BaseModel,
        step_rng: jax.Array,
        policy_obs: _model.Observation,
        critic_obs: Any,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        step_rng, noise_rng = jax.random.split(step_rng)
        noise = jax.random.normal(
            noise_rng, (policy_obs.state.shape[0], model.action_horizon, model.action_dim)
        )
        actions, log_probs = model.sample_actions(
            policy_obs,
            noise=noise,
            num_steps=num_steps,
            rng=step_rng,
            eta=eta,
            return_log_probs=True,
        )
        log_prob = jnp.sum(log_probs, axis=-1)

        if hasattr(critic, "batch_stats") and critic.batch_stats is not None:
            qs = critic.apply_fn(
                {"params": critic.params, "batch_stats": critic.batch_stats},
                critic_obs,
                actions,
            )
        else:
            qs = critic.apply_fn({"params": critic.params}, critic_obs, actions)

        if critic_reduction == "min":
            q = qs.min(axis=0)
        elif critic_reduction == "mean":
            q = qs.mean(axis=0)
        else:
            raise ValueError(f"Invalid critic reduction: {critic_reduction}")

        adv = q
        if normalize_adv:
            adv = (adv - jnp.mean(adv)) / jnp.maximum(jnp.std(adv), 1e-6)
        if adv_clip is not None:
            adv = jnp.clip(adv, -adv_clip, adv_clip)

        loss = -jnp.mean(jax.lax.stop_gradient(adv) * log_prob)

        info = {
            "loss": loss,
            "q_mean": jnp.mean(q),
            "adv_mean": jnp.mean(adv),
            "log_prob_mean": jnp.mean(log_prob),
        }

        # --- Q-guided sampling + distillation ---
        # cand_rng, loss_rng = jax.random.split(step_rng)
        # cand_noise = jax.random.normal(
        #     cand_rng,
        #     (policy_obs.state.shape[0], num_candidates, model.action_horizon, model.action_dim),
        # )
        # cand_actions, _ = jax.vmap(
        #     lambda n: model.sample_actions(
        #         policy_obs, noise=n, num_steps=num_steps, rng=loss_rng, eta=eta, return_log_probs=False
        #     )
        # )(cand_noise)
        # cand_qs = critic.apply_fn({"params": critic.params}, critic_obs, cand_actions)
        # cand_q = cand_qs.min(axis=0)
        # weights = jax.nn.softmax(cand_q / q_temperature, axis=0)
        # bc_loss = jnp.mean(weights * jnp.mean(model.compute_loss(loss_rng, policy_obs, cand_actions), axis=-1))
        # loss = bc_loss

        return loss, info

    diff_state = nnx.DiffState(0, trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, rng, policy_observation, critic_observation
    )

    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=nnx.state(model),
        opt_state=new_opt_state,
    )
    return new_state, info
