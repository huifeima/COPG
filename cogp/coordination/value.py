"""Cardinality-independent value normalization and initialization."""
from __future__ import annotations
import math
import numpy as np
from torch import nn
class RunningValueNorm:
    """Running mean/std for unnormalized value targets."""

    def __init__(self, eps: float = 1e-5):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(eps)

    @property
    def std(self) -> float:
        return float(math.sqrt(max(self.var, 1e-8)))

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        bmean = float(x.mean())
        bvar = float(x.var())
        bcount = float(x.size)
        delta = bmean - self.mean
        total = self.count + bcount
        new_mean = self.mean + delta * bcount / total
        m_a = self.var * self.count
        m_b = bvar * bcount
        m2 = m_a + m_b + delta * delta * self.count * bcount / total
        self.mean = float(new_mean)
        self.var = float(m2 / total)
        self.count = float(total)

    def normalize_np(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / (self.std + 1e-8)).astype(np.float32)

    def denormalize_scalar(self, x: float) -> float:
        return float(x) * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = float(d.get("mean", 0.0))
        self.var = float(d.get("var", 1.0))
        self.count = float(d.get("count", 1e-5))


def _orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

