"""Simulation and coordination configuration."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class EnvConfig:
    world_size: float = 800.0
    dt: float = 0.25
    physics_substeps: int = 5
    max_steps: int = 1800
    num_sheep: int = 15
    num_dogs: int = 3
    max_sheep: int = 20
    max_dogs: int = 4
    sheep_rest_distance: float = 20.0
    sheep_social_radius: float = 42.0
    sheep_max_neighbors: int = 7
    k_repulsion: float = 8.0
    k_attraction: float = 0.55
    k_alignment: float = 0.45
    sheep_damping: float = 0.75
    sheep_max_accel: float = 18.0
    sheep_max_speed: float = 6.0
    dog_max_speed: float = 12.0
    dog_influence_radius: float = 80.0
    k_dog: float = 30.0
    dog_force_power: float = 1.25
    dog_collision_radius: float = 18.0
    dog_collision_gain: float = 5.0
    boundary_margin: float = 35.0
    boundary_gain: float = 0.35
    goal_radius: float = 70.0
    success_hold_high_steps: int = 4
    outlier_radius: float = 75.0
    reward_goal_progress: float = 2.5
    reward_centroid_progress: float = 1.0
    penalty_dispersion: float = 0.01
    penalty_outlier: float = 0.2
    penalty_dog_collision: float = 0.25
    penalty_time: float = 0.02
    reward_success: float = 30.0
    flock_center_x: float = 400.0
    flock_center_y: float = 500.0
    goal_x: float = 400.0
    goal_y: float = 120.0
    compact_radius: float = 42.0
    density_reference_n: int = 9
    split_center_gap: float = 150.0
    split_radius_factor: float = 1.0
    initial_min_sheep_distance: float = 11.0
    dog_start_distance: float = 120.0
    dog_start_margin: float = 78.0
    dog_lateral_spacing: float = 65.0
    seed: int = 0

    def validate(self) -> None:
        if not 1 <= self.num_sheep <= self.max_sheep:
            raise ValueError(f'num_sheep must be in [1,{self.max_sheep}]')
        if not 1 <= self.num_dogs <= self.max_dogs:
            raise ValueError(f'num_dogs must be in [1,{self.max_dogs}]')
        if self.sheep_social_radius <= self.sheep_rest_distance:
            raise ValueError('sheep_social_radius must exceed sheep_rest_distance')
        if self.physics_substeps < 1:
            raise ValueError('physics_substeps must be >= 1')
        if self.density_reference_n < 1:
            raise ValueError('density_reference_n must be >= 1')

    def population_scaled_radius(self, n: int, factor: float=1.0) -> float:
        n = max(1, int(n))
        return float(self.compact_radius * factor * (n / float(self.density_reference_n)) ** 0.5)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PlannerConfig:
    option_horizon: int = 6
    drive_lookahead: float = 90.0
    collect_lookahead: float = 25.0
    rear_margin: float = 30.0
    flank_margin: float = 30.0
    flank_back_fraction: float = 0.55
    collect_offset: float = 42.0
    anchor_residual_scale: float = 20.0
    rear_offset: float = 72.0
    flank_back_offset: float = 45.0
    flank_lateral_offset: float = 85.0
    anchor_gain: float = 1.0
    basc_blend: float = 1.0
    graph_radius: float = 42.0
    dog_sheep_edge_radius: float = 120.0
    use_ss_edges: bool = True
    use_ds_edges: bool = True
    use_dd_edges: bool = True
    hidden_dim: int = 128
    node_dim: int = 96
    gnn_layers: int = 3
    actor_lr: float = 0.0003
    critic_lr: float = 0.0005
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 6
    minibatch_size: int = 256
    rollout_steps: int = 256
    offset_std_init: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
