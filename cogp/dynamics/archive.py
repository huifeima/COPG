"""Supervision-only pre-integration force recording with a bounded reservoir."""
from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any

import numpy as np

from .fit import sha256
from .models import STATE_FIELDS


class ForceArchive:
    """Record every twelfth force call; retain smallest seeded random priorities.

    The caller supplies the true, already capped acceleration during training
    data collection only. The recorder does not call any simulator force
    function and must never be attached to deployed candidate prediction.
    Shape is fixed within an archive, matching the training panel.
    """

    def __init__(self, seed: int, capacity: int = 60000, stride: int = 12):
        if capacity < 1 or stride < 1:
            raise ValueError("Archive capacity and force-call stride must be positive")
        self.seed = int(seed)
        self.capacity = int(capacity)
        self.stride = int(stride)
        self.rng = np.random.default_rng(self.seed)
        self.force_calls = 0
        self.eligible_samples = 0
        self._heap: list[tuple[float, int, dict[str, np.ndarray]]] = []
        self._shape: tuple[int, int] | None = None

    def observe(self, env: Any, clipped_acceleration: np.ndarray, episode_id: int) -> None:
        """Call after the true force is computed and before sheep integration."""
        self.force_calls += 1
        if self.force_calls % self.stride:
            return
        positions = np.asarray(env.sheep_pos)
        label = np.asarray(clipped_acceleration, dtype=np.float32)
        if label.shape != positions.shape or not np.isfinite(label).all():
            raise ValueError("Supervision must be finite acceleration with shape [N,2]")
        shape = (len(positions), len(env.dog_pos))
        if self._shape is None:
            self._shape = shape
        elif shape != self._shape:
            raise ValueError("A training archive cannot mix population sizes")
        self.eligible_samples += 1
        priority = float(self.rng.random())
        serial = self.eligible_samples
        # Negated priority makes the largest retained priority the heap root.
        if len(self._heap) >= self.capacity and priority >= -self._heap[0][0]:
            return
        record = {key: np.array(getattr(env, key), dtype=np.float32, copy=True)
                  for key in STATE_FIELDS[:-1]}
        record["social_graph"] = np.array(env.social_graph_mask(), dtype=bool, copy=True)
        record["accel"] = label.copy()
        record["episode_id"] = np.asarray(int(episode_id), dtype=np.int64)
        item = (-priority, serial, record)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, item)
        else:
            heapq.heapreplace(self._heap, item)

    def save(self, path: str | Path) -> dict[str, Any]:
        if not self._heap:
            raise ValueError("No eligible force samples were recorded")
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("Dynamics archive path must end in .npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        records = [item[2] for item in sorted(self._heap, key=lambda item: item[1])]
        arrays = {key: np.stack([record[key] for record in records])
                  for key in (*STATE_FIELDS, "accel", "episode_id")}
        np.savez_compressed(destination, **arrays)
        manifest = {
            "dataset_sha256": sha256(destination), "seed": self.seed,
            "label": "simulator_clipped_sheep_acceleration",
            "state_timing": "before_sheep_integration_after_shepherd_update",
            "force_calls": self.force_calls, "stride": self.stride,
            "eligible_samples": self.eligible_samples, "retained_samples": len(records),
            "capacity": self.capacity,
            "reservoir": "smallest_seeded_uniform_priorities_in_observation_order",
            "num_sheep": self._shape[0], "num_shepherds": self._shape[1],
            "episode_ids": np.unique(arrays["episode_id"]).tolist(),
        }
        destination.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n")
        return manifest
