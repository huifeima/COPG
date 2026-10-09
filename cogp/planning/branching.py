"""Clone mutable execution state while sharing frozen low-level networks."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import hashlib
import inspect
import types
import numpy as np


def _update_hash(h, value, visited):
    if value is None or isinstance(value, (bool, int, float, str, np.generic)):
        h.update(repr(value.item() if isinstance(value, np.generic) else value).encode())
        return
    if inspect.ismethod(value) or inspect.isfunction(value) or (hasattr(value, 'predict') and hasattr(value, 'policy')):
        return
    if isinstance(value, np.ndarray):
        h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
        return
    if isinstance(value, np.random.Generator):
        _update_hash(h, value.bit_generator.state, visited)
        return
    if id(value) in visited:
        h.update(b'alias'); return
    visited[id(value)] = value  # Hold temporary RNG mappings alive: Python must not reuse their IDs.
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            if str(key) in {'collect_model', 'drive_model'}:
                continue  # Frozen learned networks; hashes are checked as assets.
            h.update(str(key).encode()); _update_hash(h, value[key], visited)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _update_hash(h, item, visited)
    elif isinstance(value, (set, frozenset)):
        for item in sorted(value, key=repr):
            _update_hash(h, item, visited)
    elif hasattr(value, '__dict__'):
        h.update(type(value).__name__.encode())
        _update_hash(h, vars(value), visited)
    else:
        h.update(type(value).__name__.encode())


def state_digest(env, coordinator, context):
    h = hashlib.sha256()
    _update_hash(h, (env, coordinator, context), {})
    return h.hexdigest()


def clone_branch(env, coordinator, context):
    memo = {}
    low = coordinator.executor.low_level
    for name in ('collect_model', 'drive_model'):
        model = getattr(low, name, None)
        if model is not None:
            memo[id(model)] = model
    clone = copy.deepcopy((env, coordinator, context), memo)
    if clone[0] is env or clone[1] is coordinator or clone[1].executor.low_level is low:
        raise RuntimeError('Prediction state was not isolated')
    return clone


@contextmanager
def dynamics_context(env, predictor):
    """Replace only acceleration. None is reserved for true-simulator diagnostics."""
    if predictor is None:
        yield
        return
    existed = '_compute_sheep_forces' in vars(env)
    original = vars(env).get('_compute_sheep_forces')

    def predicted(local):
        acceleration = np.asarray(predictor.predict_numpy(local), dtype=np.float64)
        if acceleration.shape != (local.num_sheep, 2) or not np.isfinite(acceleration).all():
            raise RuntimeError('Dynamics returned invalid sheep acceleration')
        norms = np.linalg.norm(acceleration, axis=1)
        acceleration *= np.minimum(1., float(local.cfg.sheep_max_accel) / np.maximum(norms, 1e-12))[:, None]
        graph = local.social_graph_mask()
        direct = np.linalg.norm(local.dog_pos[:, None, :] - local.sheep_pos[None, :, :], axis=-1) < local.cfg.dog_influence_radius
        local._last_direct_dog_mask = direct
        local._last_graph_mask = graph
        local._last_repulsive_load = np.zeros(local.num_sheep, dtype=np.float32)
        return acceleration, local._last_repulsive_load, graph

    env._compute_sheep_forces = types.MethodType(predicted, env)
    try:
        yield
    finally:
        if existed:
            env._compute_sheep_forces = original
        else:
            del env._compute_sheep_forces


@contextmanager
def forbid_true_forces(env):
    """Any true-force access in deployed selection is an error."""
    from unittest.mock import patch
    with patch.object(type(env), '_compute_sheep_forces', side_effect=RuntimeError('True force oracle during deployed selection')):
        yield


@dataclass
class Rollout:
    env: object
    co: object
    ctx: object
    reward: float
    raw: float
    duration: int
    primitive_steps: int
    done: bool
    info: dict


def rollout_branch(env, co, ctx, spec, predictor, max_high_steps=64, memory='full'):
    before = state_digest(env, co, ctx)
    local_env, local_co, local_ctx = clone_branch(env, co, ctx)
    if memory != 'full':
        from .memory import reinitialize_history
        reinitialize_history(local_co.executor.low_level, local_env.num_dogs)
    local_co.max_option_high_steps = int(max_high_steps)
    start = int(local_env.primitive_step)
    with dynamics_context(local_env, predictor):
        reward, done, info, duration, raw, pending = local_co.step_committed(local_env, local_ctx, spec)
    terminal = bool(done or local_env.primitive_step >= local_env.cfg.max_steps or local_env.metrics()['success'] > .5)
    endpoint_ctx = pending if pending is not None else local_ctx if terminal else local_co.prepare(local_env)
    if state_digest(env, co, ctx) != before:
        raise RuntimeError('Prediction mutated the live state')
    return Rollout(local_env, local_co, endpoint_ctx, float(reward), float(raw), int(duration),
                   int(local_env.primitive_step)-start, terminal, dict(info))

