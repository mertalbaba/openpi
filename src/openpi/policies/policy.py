from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import flax.nnx as nnx
from flax.core import frozen_dict
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _mutable_tree_copy(tree: Any) -> Any:
    """Return a mutable copy of nested mappings, unfreezing FrozenDict trees."""
    if isinstance(tree, frozen_dict.FrozenDict):
        tree = frozen_dict.unfreeze(tree)
    if isinstance(tree, dict):
        return {k: _mutable_tree_copy(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_mutable_tree_copy(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_mutable_tree_copy(v) for v in tree)
    return tree


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)

        def _sample_actions_with_model_fn(
            m: _model.BaseModel,
            *,
            observation: _model.Observation,
            noise: jnp.ndarray,
            num_steps: int | jnp.ndarray = 10,
            rng: at.KeyArrayLike | None = None,
            noise_level: float = 0.0,
            return_info_dict: bool = False,
        ):
            return m.sample_actions(
                observation=observation,
                noise=noise,
                num_steps=num_steps,
                rng=rng,
                noise_level=noise_level,
                return_info_dict=return_info_dict,
            )

        self._sample_actions_with_model = nnx.jit(
            _sample_actions_with_model_fn,
            static_argnames=("noise_level", "return_info_dict", "num_steps"),
        )
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self.action_dim = model.action_dim
        self.action_horizon = model.action_horizon
        self._get_prefix_rep = nnx_utils.module_jit(model.get_prefix_rep)
        self._get_prefix_rep_with_model = nnx.jit(lambda m, *args, **kwargs: m.get_prefix_rep(*args, **kwargs))

    def _prepare_inputs(self, obs: dict) -> tuple[dict, int]:
        # Make a copy since transformations may modify the inputs in place.
        inputs = _mutable_tree_copy(obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]
            def _add_batch_dim(x):
                return jnp.broadcast_to(
                    x[jnp.newaxis, ...],
                    (batch_size,) + x.shape
                )

            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            for key in inputs:
                if key not in ["image", "state"]:
                    inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
        else:
            batch_size = 1
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return inputs, batch_size

    def _postprocess_outputs(self, outputs: dict, batch_size: int) -> dict:
        # Unbatch and convert to np.ndarray.
        if batch_size == 1:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)

    @override
    def infer(self, obs: dict, noise: jnp.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs, batch_size = self._prepare_inputs(obs)
        # self._rng, sample_rng = jax.random.split(self._rng)
        if noise is None:
            self._rng, sample_rng = jax.random.split(self._rng)
            noise = jax.random.normal(sample_rng, (batch_size, self.action_horizon, self.action_dim))
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(_model.Observation.from_dict(inputs), noise=noise, **self._sample_kwargs),
        }

        return self._postprocess_outputs(outputs, batch_size)

    def prepare_observation(self, obs: dict) -> _model.Observation:
        inputs, _ = self._prepare_inputs(obs)
        return _model.Observation.from_dict(inputs)

    def infer_with_model(
        self,
        model: _model.BaseModel,
        obs: dict,
        *,
        noise: jnp.ndarray | None = None,
        rng: at.KeyArrayLike | None = None,
        noise_level: float = 0.0,
        num_steps: int | None = None,
        return_info_dict: bool = False,
    ) -> tuple[dict, jnp.ndarray] | dict:
        inputs, batch_size = self._prepare_inputs(obs)
        if noise is None:
            if rng is None:
                self._rng, rng = jax.random.split(self._rng)
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        sample_kwargs = dict(self._sample_kwargs)
        if num_steps is not None:
            sample_kwargs["num_steps"] = num_steps

        if return_info_dict:
            actions, extras = self._sample_actions_with_model(
                model,
                observation=_model.Observation.from_dict(inputs),
                noise=noise,
                rng=rng,
                noise_level=noise_level,
                return_info_dict=True,
                **sample_kwargs,
            )
            outputs = {"state": inputs["state"], "actions": actions}
            outputs = self._postprocess_outputs(outputs, batch_size)
            outputs['extras'] = extras
        else:
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions_with_model(
                    model,
                    observation=_model.Observation.from_dict(inputs),
                    noise=noise,
                    rng=rng,
                    noise_level=noise_level,
                    **sample_kwargs,
                ),
            }
            outputs = self._postprocess_outputs(outputs, batch_size)
        return outputs
    
    @override
    def get_prefix_rep(self, obs: dict):
        inputs = _mutable_tree_copy(obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        # add batch dim and broadcast for keys that are not "images" and "state"
        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]
            def _add_batch_dim(x):
                return jnp.broadcast_to(
                    x[jnp.newaxis, ...],
                    (batch_size,) + x.shape
                )
            for key in inputs:
                if key not in ["image", "state"]:
                    inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
        else:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return self._get_prefix_rep(_model.Observation.from_dict(inputs))

    @override
    def get_prefix_rep_with_model(self, model: _model.BaseModel,  obs: dict):
        inputs = _mutable_tree_copy(obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        # add batch dim and broadcast for keys that are not "images" and "state"
        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]

            def _add_batch_dim(x):
                return jnp.broadcast_to(
                    x[jnp.newaxis, ...],
                    (batch_size,) + x.shape
                )

            for key in inputs:
                if key not in ["image", "state"]:
                    inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
        else:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return self._get_prefix_rep_with_model(model, _model.Observation.from_dict(inputs))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
