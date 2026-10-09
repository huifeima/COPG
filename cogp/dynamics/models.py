"""Sheep-acceleration predictors on the common physical state interface.

The graph formulas are reconstructed from the paper's explicit equations.
The two physical controls retain the available source definitions. A deployed
predictor never accesses the simulator's force coefficients or force routine.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


STATE_FIELDS = ("sheep_pos", "sheep_vel", "dog_pos", "dog_vel", "social_graph")
MODEL_KINDS = ("graph", "deepsets", "boids_id", "local_pooled")
GEOMETRY_DEFAULTS = {
    "world_size": 800.0,
    "sheep_rest_distance": 20.0,
    "sheep_social_radius": 42.0,
    "dog_influence_radius": 80.0,
    "sheep_max_speed": 6.0,
    "dog_max_speed": 12.0,
    "sheep_max_accel": 18.0,
    "boundary_margin": 35.0,
}


def geometry_config(config: Any) -> dict[str, float]:
    """Whitelist geometric scales; true interaction gains are deliberately absent."""
    if config is None:
        source = {}
    elif isinstance(config, Mapping):
        source = dict(config)
    elif hasattr(config, "to_dict"):
        source = config.to_dict()
    elif is_dataclass(config):
        source = asdict(config)
    else:
        source = vars(config)
    result = {key: float(source.get(key, default))
              for key, default in GEOMETRY_DEFAULTS.items()}
    if any(not np.isfinite(value) or value <= 0 for value in result.values()):
        raise ValueError("Geometry scales and physical limits must be finite and positive")
    if result["sheep_social_radius"] <= result["sheep_rest_distance"]:
        raise ValueError("Social radius must exceed sheep rest distance")
    return result


def smooth_norm(value: Tensor) -> Tensor:
    return torch.sqrt(value.square().sum(dim=-1, keepdim=True) + 1e-12)


def mlp(inputs: int, width: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU(), nn.Linear(width, outputs))


def small_output(module: nn.Sequential) -> None:
    nn.init.normal_(module[-1].weight, std=0.003)
    nn.init.zeros_(module[-1].bias)


class AccelerationModel(nn.Module):
    """Shared batching, normalization and physical output limit."""

    def __init__(self, config: Any, kind: str):
        super().__init__()
        self.config = geometry_config(config)
        self.kind = kind
        for key, value in self.config.items():
            setattr(self, key, value)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def validate_batch(batch: Mapping[str, Tensor]) -> None:
        if set(STATE_FIELDS) - set(batch):
            raise ValueError("Missing physical dynamics state fields")
        p, v, dogs, dogv, graph = (batch[key] for key in STATE_FIELDS)
        if p.ndim != 3 or p.shape[-1] != 2 or p.shape[1] < 1 or v.shape != p.shape:
            raise ValueError("Sheep positions and velocities must have shape [B,N>=1,2]")
        if (dogs.ndim != 3 or dogs.shape[0] != p.shape[0] or dogs.shape[-1] != 2
                or dogv.shape != dogs.shape):
            raise ValueError("Shepherd positions and velocities must have shape [B,M,2]")
        if graph.shape != (p.shape[0], p.shape[1], p.shape[1]):
            raise ValueError("social_graph must have shape [B,N,N]")

    def boundary(self, position: Tensor) -> tuple[Tensor, Tensor]:
        low = F.relu((self.boundary_margin - position) / self.boundary_margin)
        high = F.relu((position - (self.world_size - self.boundary_margin))
                      / self.boundary_margin)
        return torch.cat((low, high), dim=-1), low - high

    def local_geometry(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        p, v, dogs, dogv, graph = (batch[key] for key in STATE_FIELDS)
        n = p.shape[1]
        delta = p[:, :, None, :] - p[:, None, :, :]
        distance = smooth_norm(delta)
        adjacency = graph.to(dtype=p.dtype).masked_fill(
            torch.eye(n, dtype=torch.bool, device=p.device)[None], 0)
        adjacency = adjacency * (distance[..., 0] < self.sheep_social_radius).to(p.dtype)
        degree = adjacency.sum(dim=-1, keepdim=True)
        dog_delta = p[:, :, None, :] - dogs[:, None, :, :]
        dog_distance = smooth_norm(dog_delta)
        return {
            "distance": distance,
            "direction": delta / distance,
            "adjacency": adjacency,
            "degree": degree,
            "inv_degree": degree.clamp_min(1.0).reciprocal(),
            "rel_v": (v[:, None, :, :] - v[:, :, None, :]) / self.sheep_max_speed,
            "dog_distance": dog_distance,
            "dog_direction": dog_delta / dog_distance,
            "dog_rel_v": (dogv[:, None, :, :] - v[:, :, None, :]) / self.dog_max_speed,
            "proximity": F.relu(1.0 - dog_distance / self.dog_influence_radius),
        }

    def local_features(self, batch: Mapping[str, Tensor],
                       geometry: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        p, v, dogs, dogv, _ = (batch[key] for key in STATE_FIELDS)
        b, n, _ = p.shape
        m = dogs.shape[1]
        distance, direction, rel_v = (geometry[key] for key in
                                       ("distance", "direction", "rel_v"))
        speed = smooth_norm(v) / self.sheep_max_speed
        inverse = geometry["inv_degree"]
        ss = torch.cat((
            distance / self.sheep_rest_distance,
            distance / self.sheep_social_radius,
            F.relu(1.0 - distance / self.sheep_rest_distance),
            F.relu(distance / self.sheep_rest_distance - 1.0),
            (rel_v * direction).sum(-1, keepdim=True), smooth_norm(rel_v),
            speed[:, :, None, :].expand(b, n, n, 1),
            speed[:, None, :, :].expand(b, n, n, 1),
            inverse[:, :, None, :].expand(b, n, n, 1),
            inverse[:, None, :, :].expand(b, n, n, 1),
        ), dim=-1)
        dog_direction, dog_rel_v, proximity = (geometry[key] for key in
                                               ("dog_direction", "dog_rel_v", "proximity"))
        dog_speed = smooth_norm(dogv) / self.dog_max_speed
        ds = torch.cat((
            geometry["dog_distance"] / self.dog_influence_radius,
            proximity, proximity.square(),
            (dog_rel_v * dog_direction).sum(-1, keepdim=True), smooth_norm(dog_rel_v),
            speed[:, :, None, :].expand(b, n, m, 1),
            dog_speed[:, None, :, :].expand(b, n, m, 1),
            (v[:, :, None, :] / self.sheep_max_speed * dog_direction).sum(-1, keepdim=True),
            (dogv[:, None, :, :] / self.dog_max_speed * dog_direction).sum(-1, keepdim=True),
            torch.ones_like(geometry["dog_distance"]),
        ), dim=-1)
        boundary_features, _ = self.boundary(p)
        node = torch.cat((speed, boundary_features, torch.ones_like(speed)), dim=-1)
        return ss, ds, node

    def node_force(self, node_coefficients: Tensor, batch: Mapping[str, Tensor]) -> Tensor:
        _, restoring = self.boundary(batch["sheep_pos"])
        return (-F.softplus(node_coefficients[..., :1]) * batch["sheep_vel"]
                + F.softplus(node_coefficients[..., 1:]) * restoring)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        self.validate_batch(batch)
        raw = self.raw_acceleration(batch)
        return raw / (smooth_norm(raw) / self.sheep_max_accel).clamp_min(1.0)

    def raw_acceleration(self, batch: Mapping[str, Tensor]) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict_numpy(self, env: Any) -> np.ndarray:
        """Rebuild the social mask on every call, including each physical substep."""
        parameter = next(self.parameters())
        values = {key: np.array(getattr(env, key), copy=True) for key in STATE_FIELDS[:-1]}
        values["social_graph"] = np.array(env.social_graph_mask(), copy=True)
        batch = {key: torch.as_tensor(value, device=parameter.device,
                 dtype=torch.bool if key == "social_graph" else parameter.dtype).unsqueeze(0)
                 for key, value in values.items()}
        return self(batch)[0].cpu().numpy().astype(np.float32, copy=False)


class GraphDynamics(AccelerationModel):
    """Edge-conditioned local physical model: 14,726 parameters."""

    def __init__(self, config: Any):
        super().__init__(config, "graph")
        self.ss_coefficients = mlp(10, 64, 2)
        self.ds_coefficients = mlp(10, 64, 2)
        self.node_terms = mlp(6, 64, 2)
        for module in (self.ss_coefficients, self.ds_coefficients, self.node_terms):
            small_output(module)

    def raw_acceleration(self, batch: Mapping[str, Tensor]) -> Tensor:
        geometry = self.local_geometry(batch)
        ss_features, ds_features, node_features = self.local_features(batch, geometry)
        ss_coefficients = self.ss_coefficients(ss_features)
        ds_coefficients = self.ds_coefficients(ds_features)
        ss_mask = geometry["adjacency"][..., None]
        sheep_force = self.sheep_max_accel * (
            (ss_mask * ss_coefficients[..., :1] * geometry["direction"]).sum(2)
            + (ss_mask * ss_coefficients[..., 1:] * geometry["rel_v"]).sum(2)
            * geometry["inv_degree"])
        shepherd_force = self.sheep_max_accel * (
            geometry["proximity"] *
            (ds_coefficients[..., :1] * geometry["dog_direction"]
             + ds_coefficients[..., 1:] * geometry["dog_rel_v"])).sum(2)
        return sheep_force + shepherd_force + self.node_force(self.node_terms(node_features), batch)


class DeepSetsDynamics(AccelerationModel):
    """Global set predictor reconstructed from the specified inputs/count (13,890).

    This does not claim original-weight parity: the retained material specifies
    pooling, feature groups and count, but not the original layer source.
    """

    def __init__(self, config: Any):
        super().__init__(config, "deepsets")
        self.sheep_encoder = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        self.shepherd_encoder = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        self.decoder = mlp(74, 64, 2)
        small_output(self.decoder)

    def raw_acceleration(self, batch: Mapping[str, Tensor]) -> Tensor:
        p, v, dogs, dogv, _ = (batch[key] for key in STATE_FIELDS)
        b, n, _ = p.shape
        m = dogs.shape[1]
        sheep = torch.cat((2 * p / self.world_size - 1, v / self.sheep_max_speed), dim=-1)
        shepherds = torch.cat((2 * dogs / self.world_size - 1, dogv / self.dog_max_speed), dim=-1)
        sheep_mean = self.sheep_encoder(sheep).mean(dim=1, keepdim=True).expand(b, n, 32)
        if m:
            dog_mean = self.shepherd_encoder(shepherds).mean(dim=1, keepdim=True).expand(b, n, 32)
        else:
            dog_mean = p.new_zeros(b, n, 32)
        counts = p.new_tensor([np.log1p(n), np.log1p(m)]).view(1, 1, 2).expand(b, n, 2)
        boundary_features, _ = self.boundary(p)
        return self.sheep_max_accel * self.decoder(
            torch.cat((sheep, sheep_mean, dog_mean, counts, boundary_features), dim=-1))


class BoidsID(AccelerationModel):
    """Seven fitted positive parameters; no actual simulator gains are read."""

    PARAMETER_NAMES = ("repulsion", "attraction", "alignment", "dog_repulsion",
                       "damping", "boundary", "dog_distance_exponent")

    def __init__(self, config: Any):
        super().__init__(config, "boids_id")
        self.log_parameters = nn.Parameter(torch.zeros(7))
        self.register_buffer("gain_scales", torch.tensor([
            self.sheep_max_accel / self.sheep_rest_distance,
            self.sheep_max_accel / self.sheep_rest_distance,
            self.sheep_max_accel / self.sheep_max_speed,
            self.sheep_max_accel,
            self.sheep_max_accel / self.sheep_max_speed,
            1.0, 1.0,
        ]))

    def positive_parameters(self) -> Tensor:
        return self.gain_scales * self.log_parameters.clamp(-12.0, 12.0).exp()

    @torch.no_grad()
    def coefficient_report(self) -> dict[str, float]:
        return dict(zip(self.PARAMETER_NAMES, self.positive_parameters().cpu().tolist()))

    def raw_acceleration(self, batch: Mapping[str, Tensor]) -> Tensor:
        g = self.local_geometry(batch)
        k_rep, k_att, k_ali, k_dog, damping, boundary_gain, exponent = self.positive_parameters().unbind()
        mask = g["adjacency"][..., None]
        repulsion = (mask * F.relu(self.sheep_rest_distance - g["distance"]) * g["direction"]).sum(2)
        attraction = -(mask * F.relu(g["distance"] - self.sheep_rest_distance) * g["direction"]).sum(2)
        alignment = (mask * g["rel_v"] * self.sheep_max_speed).sum(2) * g["inv_degree"]
        proximity = g["proximity"]
        power = (proximity.clamp_min(1e-12).log() * exponent).exp() * (proximity > 0)
        dog_repulsion = (power * g["dog_direction"]).sum(2)
        _, boundary = self.boundary(batch["sheep_pos"])
        return (k_rep * repulsion + k_att * attraction + k_ali * alignment
                + k_dog * dog_repulsion - damping * batch["sheep_vel"] + boundary_gain * boundary)


class LocalPooled(AccelerationModel):
    """Receiver-level pooled coefficients on the same physical bases (14,758)."""

    def __init__(self, config: Any):
        super().__init__(config, "local_pooled")
        self.ss_encoder = mlp(10, 48, 16)
        self.ds_encoder = mlp(10, 48, 16)
        self.ss_decoder = nn.Sequential(nn.Linear(18, 64), nn.SiLU(), nn.Linear(64, 2))
        self.ds_decoder = nn.Sequential(nn.Linear(18, 64), nn.SiLU(), nn.Linear(64, 2))
        self.node_terms = mlp(6, 64, 2)
        for module in (self.ss_decoder, self.ds_decoder, self.node_terms):
            small_output(module)

    @staticmethod
    def pool(encoded: Tensor, weight: Tensor) -> Tensor:
        mass = weight.sum(dim=2)
        count = (weight > 0).to(encoded.dtype).sum(dim=2)
        mean = (encoded * weight).sum(dim=2) / mass.clamp_min(1e-12)
        return torch.cat((mean, count.log1p(), mass.log1p()), dim=-1)

    def raw_acceleration(self, batch: Mapping[str, Tensor]) -> Tensor:
        g = self.local_geometry(batch)
        ss_features, ds_features, node_features = self.local_features(batch, g)
        ss_weight = g["adjacency"][..., None]
        ss_coefficients = self.ss_decoder(self.pool(self.ss_encoder(ss_features), ss_weight))
        ds_coefficients = self.ds_decoder(self.pool(self.ds_encoder(ds_features), g["proximity"]))
        sheep_force = self.sheep_max_accel * (
            ss_coefficients[..., :1] * (ss_weight * g["direction"]).sum(2)
            + ss_coefficients[..., 1:] * (ss_weight * g["rel_v"]).sum(2) * g["inv_degree"])
        shepherd_force = self.sheep_max_accel * (
            ds_coefficients[..., :1] * (g["proximity"] * g["dog_direction"]).sum(2)
            + ds_coefficients[..., 1:] * (g["proximity"] * g["dog_rel_v"]).sum(2))
        node_coefficients = self.node_terms(node_features)
        _, boundary = self.boundary(batch["sheep_pos"])
        # Preserve the available control's arithmetic order as well as its formula.
        return (sheep_force + shepherd_force
                - F.softplus(node_coefficients[..., :1]) * batch["sheep_vel"]
                + F.softplus(node_coefficients[..., 1:]) * boundary)


def build_model(kind: str, envcfg: Any = None) -> AccelerationModel:
    constructors = {"graph": GraphDynamics, "deepsets": DeepSetsDynamics,
                    "boids_id": BoidsID, "local_pooled": LocalPooled}
    if kind not in constructors:
        raise ValueError(f"Unknown learned dynamics kind {kind!r}; expected {MODEL_KINDS}")
    return constructors[kind](envcfg)
