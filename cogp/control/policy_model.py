"""Masked set feature extractor used by the frozen BASC policies."""
from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class BASCWPDeepSetsExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, per_boundary_dim: int=12, global_dim: int=14, embed_dim: int=64, features_dim: int=128, phi_layers: Tuple[int, ...]=(64, 64), rho_layers: Tuple[int, ...]=(128,)):
        super().__init__(observation_space, features_dim)
        layers = []
        d = int(per_boundary_dim)
        for h in phi_layers:
            layers += [nn.Linear(d, int(h)), nn.ReLU()]
            d = int(h)
        layers += [nn.Linear(d, int(embed_dim)), nn.ReLU()]
        self.phi = nn.Sequential(*layers)
        layers = []
        d = int(embed_dim) + int(global_dim)
        for h in rho_layers:
            layers += [nn.Linear(d, int(h)), nn.ReLU()]
            d = int(h)
        layers += [nn.Linear(d, int(features_dim)), nn.ReLU()]
        self.rho = nn.Sequential(*layers)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = obs['boundary_set'].float()
        m = obs['boundary_mask'].float().unsqueeze(-1)
        g = obs['global'].float()
        e = self.phi(x) * m
        pooled = e.sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.rho(torch.cat([pooled, g], dim=-1))
