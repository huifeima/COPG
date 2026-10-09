"""Load original frozen policies without imports of historical project modules."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


def checkpoint_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file() and path.suffix != '.zip':
        path = path.with_suffix('.zip')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def low_level_policy_kwargs() -> dict:
    from cogp.control.policy_model import BASCWPDeepSetsExtractor
    return dict(
        features_extractor_class=BASCWPDeepSetsExtractor,
        features_extractor_kwargs=dict(
            per_boundary_dim=12, global_dim=14, embed_dim=64, features_dim=128,
        ),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )


def load_frozen_policy(path: str | Path, *, device: str = 'auto'):
    """Use the same policy network with a clean feature-extractor class path.

    SB3 replaces the serialized policy_kwargs before cloudpickle decoding that
    field. Parameter loading remains strict; incompatible weights fail rather
    than being silently substituted. A supplied original policy archive is
    required. The bundled engineering tests never supply a scientific policy.
    """
    resolved=Path(path).expanduser().resolve()
    if not resolved.is_file() and resolved.suffix!='.zip':resolved=resolved.with_suffix('.zip')
    return _load_frozen(str(resolved),str(device),checkpoint_sha256(resolved))


@lru_cache(maxsize=6)
def _load_frozen(path,device,checksum):
    from stable_baselines3 import PPO
    kwargs = low_level_policy_kwargs()
    model = PPO.load(str(path), device=device,
                     custom_objects={'policy_kwargs': kwargs})
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    if checkpoint_sha256(path)!=checksum:raise RuntimeError('Frozen BASC checkpoint changed during loading')
    model.cogp_checkpoint_sha256 = checksum
    return model
