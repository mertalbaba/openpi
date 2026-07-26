"""Transforms + weight loader for the SONIC token-predicting VLA (fully-latent pi0.5).

A training sample / inference observation is the fully-latent format:
    ego frame + instruction + previous SONIC tokens  ->  target token chunk [horizon, 64]
No proprioception. The action space is the 64-d SONIC FSQ token (this project's target);
`state` is a zero placeholder, kept only because PadStatesAndActions requires it (pi0.5 with
discrete_state_input=False never reads it).
"""
import dataclasses

import flax.traverse_util
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import download
from openpi.training import weight_loaders


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:  # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class SonicTokenInputs(transforms.DataTransformFn):
    """Map a SonicTokenDataset sample (training) or an observation (inference) to the model dict.

    Used for both training and inference, so it must accept either `target_tokens`/`target_valid`
    (training, from the dataset) or neither (inference). The ego frame goes to `base_0_rgb`; the
    two wrist slots are zero-padded + masked (single-camera embodiment)."""

    # Model action dimension (= SONIC token dim, 64). Only used to size the dummy state.
    action_dim: int = 64
    # State width: 32 (q_dev+gravity) or 96 (+ the 1s-lagged hand-state token, bhs configs).
    # Only sizes the zeros fallback when no state is provided.
    state_dim: int = 32

    def __call__(self, data: dict) -> dict:
        image = _parse_image(data["image"])
        zeros_img = np.zeros_like(image)
        inputs = {
            # Proprio state (q_dev(29) + gravity(3) [+ lagged hand token(64)]) when a proprio source
            # is present; used by pi0.5 only if discrete_state_input=True (tokenized into the
            # prompt). Fully-latent configs still receive it but the model ignores it. Fallback = zeros.
            "state": np.asarray(data.get("state", np.zeros(self.state_dim, np.float32)), dtype=np.float32),
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": zeros_img,
                "right_wrist_0_rgb": zeros_img,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        # Previous-token history (raw FSQ tokens; projected into the prefix by the model).
        # Zero-history configs (prev_token_history=0, the leak-free variant) emit an empty
        # (0, 64) array -- omit the key so Observation.prev_tokens stays None and the model
        # takes the vanilla pi0.5 path (no prev_token_proj exists in that case).
        prev = np.asarray(data.get("prev_tokens", np.zeros((0, 64))), dtype=np.float32)
        if prev.shape[0] > 0:
            inputs["prev_tokens"] = prev
        # Targets are only present during training.
        if "target_tokens" in data:
            inputs["actions"] = np.asarray(data["target_tokens"], dtype=np.float32)
        if "target_valid" in data:
            inputs["action_valid"] = np.asarray(data["target_valid"], dtype=bool)
        # Body+hand mode: per-(timestep, dim) validity (hand dims of hand-less corpora are
        # loss-masked). Absent for all 64-dim configs -> Observation.action_dim_valid stays None.
        if "target_dim_valid" in data:
            inputs["action_dim_valid"] = np.asarray(data["target_dim_valid"], dtype=bool)
        # Language: dataset uses "instruction"; inference uses "prompt".
        instr = data.get("instruction", data.get("prompt"))
        if instr is not None:
            inputs["prompt"] = instr
        return inputs


@dataclasses.dataclass(frozen=True)
class SonicTokenOutputs(transforms.DataTransformFn):
    """Inference: the model's `actions` ARE the SONIC token chunk [horizon, 64]; pass through.
    Decoding tokens -> 29-DOF motion happens downstream (frozen SONIC decoder), not here."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])}


@dataclasses.dataclass(frozen=True)
class SonicCheckpointWeightLoader(weight_loaders.WeightLoader):
    """Load pi0.5 base weights, keeping only params whose key AND shape match the fresh model.

    The SONIC VLA resizes the action projections (32 -> 64 dim) and adds `prev_token_proj`,
    so those params do not match the pi05_base checkpoint. This loader transplants the matching
    pretrained weights (PaliGemma LLM + SigLIP vision + time MLPs) and leaves the resized / new
    params at their fresh initialization."""

    params_path: str

    def load(self, params):
        print(f"[SonicCheckpointWeightLoader] downloading + loading base weights from "
              f"{self.params_path} (first run fetches several GB)...", flush=True)
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")
        result, kept, fresh = {}, 0, 0
        for k, ref_v in flat_ref.items():
            lv = flat_loaded.get(k)
            if lv is not None and tuple(lv.shape) == tuple(ref_v.shape):
                result[k] = lv.astype(ref_v.dtype) if lv.dtype != ref_v.dtype else lv
                kept += 1
            else:
                result[k] = ref_v  # resized (action proj) or new (prev_token_proj) -> fresh init
                fresh += 1
        print(f"[SonicCheckpointWeightLoader] transplanted {kept} params, fresh-init {fresh}")
        return flax.traverse_util.unflatten_dict(result, sep="/")
