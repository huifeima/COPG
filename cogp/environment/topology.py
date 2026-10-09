"""Deterministic topology utilities for acquisition and phase gates."""
import numpy as np
from typing import List

def connected_components(graph: np.ndarray) -> List[np.ndarray]:
    graph = np.asarray(graph, dtype=bool)
    n = len(graph)
    seen = np.zeros(n, dtype=bool)
    comps: List[np.ndarray] = []
    for root in range(n):
        if seen[root]:
            continue
        stack = [root]
        seen[root] = True
        comp = []
        while stack:
            i = stack.pop()
            comp.append(int(i))
            for j in np.where(graph[i])[0]:
                j = int(j)
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
        comps.append(np.asarray(comp, dtype=np.int64))
    return sorted(comps, key=len, reverse=True)

def graph_lcc_fraction(env) -> float:
    comps = connected_components(env.social_graph_mask())
    if not comps:
        return 1.0
    return float(len(comps[0]) / max(1, int(env.num_sheep)))

def collect_candidates(env) -> np.ndarray:
    p = env.sheep_pos.astype(np.float64)
    comps = connected_components(env.social_graph_mask())
    candidates: List[int] = []
    if len(comps) > 1:
        main = comps[0]
        main_c = p[main].mean(axis=0)
        for comp in comps[1:]:
            d = np.linalg.norm(p[comp] - main_c[None, :], axis=1)
            candidates.append(int(comp[int(np.argmax(d))]))
    if not candidates:
        c = env.flock_center().astype(np.float64)
        out_idx = np.where(env.outlier_mask())[0]
        if len(out_idx):
            radial = np.linalg.norm(p[out_idx] - c[None, :], axis=1)
            candidates = [int(x) for x in out_idx[np.argsort(-radial)]]
    return np.asarray(candidates, dtype=np.int64)
