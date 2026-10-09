"""Shared padded physical observations."""

from __future__ import annotations
from typing import Dict, List
import numpy as np
from cogp.environment.geometry import convex_hull_indices
SHEEP_FEAT_DIM = 14
DOG_FEAT_DIM = 10
GLOBAL_FEAT_DIM = 12

class GraphObservationBuilder:

    def __init__(self, max_sheep: int=20, max_dogs: int=4, graph_radius: float=42.0, dog_sheep_edge_radius: float=120.0, use_ss_edges: bool=True, use_ds_edges: bool=True, use_dd_edges: bool=True):
        self.max_sheep = int(max_sheep)
        self.max_dogs = int(max_dogs)
        self.graph_radius = float(graph_radius)
        self.dog_sheep_edge_radius = float(dog_sheep_edge_radius)
        self.use_ss_edges = bool(use_ss_edges)
        self.use_ds_edges = bool(use_ds_edges)
        self.use_dd_edges = bool(use_dd_edges)

    def build_all_agents(self, env, prev_roles: np.ndarray | None=None) -> List[Dict[str, np.ndarray]]:
        if prev_roles is None:
            prev_roles = np.full(env.num_dogs, -1, dtype=np.int64)
        return [self.build_agent(env, j, prev_roles) for j in range(env.num_dogs)]

    def build_agent(self, env, self_dog: int, prev_roles: np.ndarray) -> Dict[str, np.ndarray]:
        n, m = (env.num_sheep, env.num_dogs)
        scale = float(env.cfg.world_size)
        c = env.flock_center().astype(np.float64)
        vf = env.flock_velocity().astype(np.float64)
        goal = env.goal.astype(np.float64)
        hull = set(map(int, convex_hull_indices(env.sheep_pos).tolist()))
        graph = env.social_graph_mask()
        degree = graph.sum(axis=1)
        radial = np.linalg.norm(env.sheep_pos.astype(np.float64) - c[None, :], axis=1)
        outlier = env.outlier_mask().astype(np.float32)
        sf = np.zeros((self.max_sheep, SHEEP_FEAT_DIM), dtype=np.float32)
        sm = np.zeros(self.max_sheep, dtype=np.float32)
        dog_self_pos = env.dog_pos[self_dog].astype(np.float64)
        for i in range(n):
            p = env.sheep_pos[i].astype(np.float64)
            v = env.sheep_vel[i].astype(np.float64)
            rel_c = (p - c) / scale
            rel_v = (v - vf) / max(env.cfg.sheep_max_speed, 1e-08)
            rel_g = (goal - p) / scale
            rel_d = (dog_self_pos - p) / scale
            nbr = np.where(graph[i])[0]
            mean_nbr_dist = 0.0 if nbr.size == 0 else float(np.mean(np.linalg.norm(env.sheep_pos[nbr] - env.sheep_pos[i], axis=1))) / max(env.cfg.sheep_social_radius, 1e-08)
            sf[i] = np.asarray([rel_c[0], rel_c[1], rel_v[0], rel_v[1], rel_g[0], rel_g[1], rel_d[0], rel_d[1], float(degree[i]) / max(env.cfg.sheep_max_neighbors * 2, 1), mean_nbr_dist, 1.0 if i in hull else 0.0, outlier[i], radial[i] / scale, float(np.linalg.norm(v)) / max(env.cfg.sheep_max_speed, 1e-08)], dtype=np.float32)
            sm[i] = 1.0
        df = np.zeros((self.max_dogs, DOG_FEAT_DIM), dtype=np.float32)
        dm = np.zeros(self.max_dogs, dtype=np.float32)
        for j in range(m):
            p = env.dog_pos[j].astype(np.float64)
            v = env.dog_vel[j].astype(np.float64)
            rel_c = (p - c) / scale
            rel_g = (goal - p) / scale
            rel_self = (p - dog_self_pos) / scale
            prev = int(prev_roles[j]) if j < len(prev_roles) else -1
            prev_norm = -1.0 if prev < 0 else prev / 3.0
            df[j] = np.asarray([rel_c[0], rel_c[1], v[0] / max(env.cfg.dog_max_speed, 1e-08), v[1] / max(env.cfg.dog_max_speed, 1e-08), rel_g[0], rel_g[1], rel_self[0], rel_self[1], 1.0 if j == self_dog else 0.0, prev_norm], dtype=np.float32)
            dm[j] = 1.0
        ss = np.zeros((self.max_sheep, self.max_sheep), dtype=np.float32)
        if self.use_ss_edges:
            ss[:n, :n] = graph.astype(np.float32)
        ds = np.zeros((self.max_dogs, self.max_sheep), dtype=np.float32)
        dmat = np.linalg.norm(env.dog_pos[:, None, :] - env.sheep_pos[None, :, :], axis=-1)
        if self.use_ds_edges:
            ds[:m, :n] = (dmat <= self.dog_sheep_edge_radius).astype(np.float32)
        dd = np.zeros((self.max_dogs, self.max_dogs), dtype=np.float32)
        if self.use_dd_edges and m > 1:
            dd[:m, :m] = 1.0
            np.fill_diagonal(dd[:m, :m], 0.0)
        out_count = float(np.sum(outlier))
        G = np.asarray([(goal[0] - c[0]) / scale, (goal[1] - c[1]) / scale, env.centroid_goal_distance() / scale, env.mean_goal_distance() / scale, env.dispersion() / scale, env.max_radius() / scale, out_count / max(n, 1), float(graph.sum() / 2.0) / max(n * (n - 1) / 2.0, 1.0), float(n) / max(env.cfg.max_sheep, 1), float(m) / max(env.cfg.max_dogs, 1), float(env.actuation_overlap()), float(env.current_scenario == 'split')], dtype=np.float32)
        return {'sheep': sf, 'sheep_mask': sm, 'dogs': df, 'dog_mask': dm, 'ss_adj': ss, 'ds_adj': ds, 'dd_adj': dd, 'global': G, 'self_dog': np.asarray(self_dog, dtype=np.int64)}

    def global_state_vector(self, env) -> np.ndarray:
        scale = float(env.cfg.world_size)
        sp = np.zeros((self.max_sheep, 4), dtype=np.float32)
        sm = np.zeros(self.max_sheep, dtype=np.float32)
        dp = np.zeros((self.max_dogs, 4), dtype=np.float32)
        dm = np.zeros(self.max_dogs, dtype=np.float32)
        n, m = (env.num_sheep, env.num_dogs)
        sp[:n, :2] = env.sheep_pos / scale
        sp[:n, 2:] = env.sheep_vel / max(env.cfg.sheep_max_speed, 1e-08)
        sm[:n] = 1.0
        dp[:m, :2] = env.dog_pos / scale
        dp[:m, 2:] = env.dog_vel / max(env.cfg.dog_max_speed, 1e-08)
        dm[:m] = 1.0
        goal = env.goal.astype(np.float32) / scale
        metrics = env.metrics()
        extras = np.asarray([metrics['centroid_goal_distance'] / scale, metrics['mean_goal_distance'] / scale, metrics['dispersion'] / scale, metrics['outlier_count'] / max(n, 1)], dtype=np.float32)
        return np.concatenate([sp.reshape(-1), sm, dp.reshape(-1), dm, goal, extras], axis=0).astype(np.float32)

def stack_obs(obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = ['sheep', 'sheep_mask', 'dogs', 'dog_mask', 'ss_adj', 'ds_adj', 'dd_adj', 'global', 'self_dog']
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in keys}
