"""Canonical observation and action reflections."""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
BOUNDARY_VECTOR_PAIRS = ((0, 1), (3, 4), (6, 7), (9, 10))
GLOBAL_VECTOR_PAIRS = ((0, 1), (3, 4), (5, 6), (11, 12))

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(v))
    if n < 1e-09:
        return np.array([0.0, -1.0], dtype=np.float64)
    return v / n

def reflection_matrix_from_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    g = np.asarray(obs['global'], dtype=np.float64)
    f = _unit(g[0:2])
    return 2.0 * np.outer(f, f) - np.eye(2, dtype=np.float64)

def reflect_vec(v: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (np.asarray(R, dtype=np.float64) @ np.asarray(v, dtype=np.float64).reshape(2)).astype(np.float32)

def reflect_observation(obs: Dict[str, np.ndarray], R: np.ndarray | None=None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    if R is None:
        R = reflection_matrix_from_obs(obs)
    R = np.asarray(R, dtype=np.float64).reshape(2, 2)
    X = np.asarray(obs['boundary_set'], dtype=np.float32).copy()
    G = np.asarray(obs['global'], dtype=np.float32).copy()
    M = np.asarray(obs['boundary_mask'], dtype=np.float32).copy()
    for i, j in BOUNDARY_VECTOR_PAIRS:
        X[:, [i, j]] = (R @ X[:, [i, j]].astype(np.float64).T).T.astype(np.float32)
    for i, j in GLOBAL_VECTOR_PAIRS:
        G[[i, j]] = (R @ G[[i, j]].astype(np.float64)).astype(np.float32)
    return ({'boundary_set': X, 'boundary_mask': M, 'global': G}, R)

def reflect_action(action: np.ndarray, R: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float64).reshape(2)
    return np.clip(np.asarray(R, dtype=np.float64) @ a, -1.0, 1.0).astype(np.float32)

def involution_error(obs: Dict[str, np.ndarray]) -> float:
    o1, R = reflect_observation(obs)
    o2, _ = reflect_observation(o1, R)
    e1 = float(np.max(np.abs(np.asarray(o2['boundary_set']) - np.asarray(obs['boundary_set']))))
    e2 = float(np.max(np.abs(np.asarray(o2['global']) - np.asarray(obs['global']))))
    return max(e1, e2)
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    obs = {'boundary_set': rng.normal(size=(10, 12)).astype(np.float32), 'boundary_mask': np.ones(10, dtype=np.float32), 'global': rng.normal(size=14).astype(np.float32)}
    obs['global'][0:2] = np.array([0.2, -0.5], dtype=np.float32)
    print('involution_error', involution_error(obs))
