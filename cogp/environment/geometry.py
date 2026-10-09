"""Shared geometric primitives."""

from __future__ import annotations
import math
import numpy as np

def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))

def unit(v: np.ndarray, eps: float=1e-09) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float64)
    return v / n

def perp(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return np.array([-v[1], v[0]], dtype=np.float64)

def clip_norm(v: np.ndarray, max_norm: float, eps: float=1e-09) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < eps:
        return v
    return v * (float(max_norm) / n)

def convex_hull_indices(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    n = int(len(points))
    if n <= 2:
        return np.arange(n, dtype=np.int64)
    order = sorted(range(n), key=lambda i: (float(points[i, 0]), float(points[i, 1]), i))

    def cross(o: int, a: int, b: int) -> float:
        return float((points[a, 0] - points[o, 0]) * (points[b, 1] - points[o, 1]) - (points[a, 1] - points[o, 1]) * (points[b, 0] - points[o, 0]))
    lower = []
    for i in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0.0:
            lower.pop()
        lower.append(i)
    upper = []
    for i in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0.0:
            upper.pop()
        upper.append(i)
    hull = lower[:-1] + upper[:-1]
    if not hull:
        return np.arange(n, dtype=np.int64)
    seen, out = (set(), [])
    for i in hull:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return np.asarray(out, dtype=np.int64)

def sample_disk(rng: np.random.Generator, center: np.ndarray, radius: float) -> np.ndarray:
    r = radius * math.sqrt(float(rng.uniform()))
    a = float(rng.uniform(0.0, 2.0 * math.pi))
    return np.asarray(center, dtype=np.float64) + r * np.array([math.cos(a), math.sin(a)], dtype=np.float64)
