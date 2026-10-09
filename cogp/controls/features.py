"""Read-only execution-state features for matched information controls.

No prediction, task construction, prepare(), or environment reset is performed.
The representation is a prespecified partial state, not a sufficiency claim.
"""
from __future__ import annotations
import math
import numpy as np

FEATURE_SCHEMA = "execution-state"
GLOBAL_FEATURE_NAMES = (
    "episode_primitive_fraction", "success_hold_fraction",
    "phase_recover", "phase_drive", "previous_phase_recover", "previous_phase_drive",
    "phase_transitioned", "connected_hold_fraction", "drive_segment_active",
    "drive_pb_x", "drive_pb_y", "drive_age_fraction", "rear_barrier_drive_context",
    "commitment_active", "commitment_elapsed_checks_div64", "commitment_remaining_checks_div64",
    "commitment_budget_checks_div64", "commitment_start_recover", "commitment_start_drive",
    "segment_completions_since_start", "acquisition_ends_since_start",
)
DOG_FEATURE_NAMES = (
    "staging_active", "staging_armed", "staging_hold_fraction", "staging_target_valid",
    "staging_target_x", "staging_target_y", "staging_target_vx", "staging_target_vy",
    "rear_barrier_active", "rear_barrier_hold_fraction", "active_task_valid",
    "task_collect", "task_rear", "task_left", "task_right", "task_reserve",
    "task_target_valid", "task_target_x", "task_target_y",
    "component_center_x", "component_center_y", "component_vx", "component_vy",
    "component_size_fraction", "pressure_lateral", "pressure_rank_fraction",
    "pressure_count_div3", "reserve_rank_fraction", "execution_gain",
    "execution_offset_x", "execution_offset_y",
)


def _array_item(owner, name, index, default=0):
    value = getattr(owner, name, ())
    return value[index] if value is not None and len(value) > index else default


def executor_features_from_state(env, co, ctx, commitment=None):
    """Read current memory; no prepare, reward check, reset, or future labels.

    The schema is a documented partial execution-state representation, not a
    claim of complete Markov sufficiency. Atomic acquisition is never paused.
    Per-dog rows and membership masks support variable M,N. Invalid targets
    have a zero vector plus an explicit validity bit, never a numeric ID input.
    """
    m, n = int(env.num_dogs), int(env.num_sheep)
    scale = max(float(env.cfg.world_size), 1e-9)
    speed = max(float(env.cfg.sheep_max_speed), 1e-9)
    low, gate = co.executor.low_level, co.phase_gate
    manager = co.executor.drive_segments
    segment = manager.current
    phase = str(ctx.phase)
    previous = str(ctx.previous_phase)
    active = bool(commitment is not None and commitment.active)
    elapsed = int(commitment.elapsed) if active else 0
    budget = int(commitment.E) if active else 0
    pb = np.asarray(segment.pb, dtype=np.float64) / scale if segment is not None else np.zeros(2)
    global_row = np.asarray([
        float(env.primitive_step) / max(1, int(env.cfg.max_steps)),
        float(getattr(env, "success_hold", 0)) / max(1, int(getattr(env.cfg, "success_hold_high_steps", 4))),
        float(phase == "RECOVER"), float(phase == "DRIVE"),
        float(previous == "RECOVER"), float(previous == "DRIVE"), float(ctx.transitioned),
        float(gate.connected_hold) / max(1, int(gate.recover_exit_hold)),
        float(segment is not None), float(pb[0]), float(pb[1]),
        float(segment.age if segment is not None else 0) / max(1, int(manager.segment_steps)),
        float(getattr(low, "_rear_barrier_drive_context", False)), float(active),
        elapsed / 64.0, max(0, budget - elapsed) / 64.0, budget / 64.0,
        float(active and commitment.start_phase == "RECOVER"),
        float(active and commitment.start_phase == "DRIVE"),
        float(int(manager.completed_count) - commitment.start_seg) if active else 0.0,
        float(int(co.acquire_aborted_total + co.acquire_timeout_total) - commitment.start_abort) if active else 0.0,
    ], dtype=np.float32)
    dogs = np.zeros((m, len(DOG_FEATURE_NAMES)), dtype=np.float32)
    components = np.zeros((m, n), dtype=np.float32)
    targets = np.zeros((m, n), dtype=np.float32)
    task_by_dog = commitment.spec.tasks_by_dog if active else ()
    for j in range(m):
        target = int(_array_item(low, "_stage_target", j, -1))
        valid = 0 <= target < n
        p = np.asarray(env.sheep_pos[target]) / scale if valid else np.zeros(2)
        v = np.asarray(env.sheep_vel[target]) / speed if valid else np.zeros(2)
        row = [float(_array_item(low, "_stage_active", j)), float(_array_item(low, "_stage_armed", j)),
               float(_array_item(low, "_stage_hold", j)) / max(1, int(getattr(low, "staging_exit_hold_primitives", 2))),
               float(valid), float(p[0]), float(p[1]), float(v[0]), float(v[1]),
               float(_array_item(low, "_rear_barrier_active", j)),
               float(_array_item(low, "_rear_barrier_hold", j)) / max(1, int(getattr(low, "rear_barrier_release_hold_primitives", 2)))]
        dogs[j, :10] = row
        if j < len(task_by_dog):
            task = task_by_dog[j]
            typ = int(task.task_type)
            if typ not in range(5):
                raise ValueError("Unknown task type in active commitment")
            target = int(task.target)
            valid = 0 <= target < n and typ == 0
            if valid:
                targets[j, target] = 1
            p = np.asarray(env.sheep_pos[target]) / scale if valid else np.zeros(2)
            ids = np.asarray(task.component_members, dtype=np.int64)
            if ids.size and (np.any(ids < 0) or np.any(ids >= n)):
                raise ValueError("Active assignment has invalid component membership")
            if ids.size:
                components[j, ids] = 1
                center = np.asarray(env.sheep_pos)[ids].mean(0) / scale
                velocity = np.asarray(env.sheep_vel)[ids].mean(0) / speed
            else:
                center, velocity = np.zeros(2), np.zeros(2)
            offsets = np.asarray(commitment.exe.action.offsets[j], dtype=float)
            dogs[j, 10:] = [1.0, *[float(typ == k) for k in range(5)], float(valid),
                            float(p[0]), float(p[1]), float(center[0]), float(center[1]),
                            float(velocity[0]), float(velocity[1]), len(ids) / max(1, n),
                            float(task.pressure_lateral), float(task.pressure_rank) / max(1, int(task.pressure_count)),
                            float(task.pressure_count) / 3.0, float(task.reserve_rank) / max(1, m),
                            float(commitment.exe.action_gains[j]), float(offsets[0]), float(offsets[1])]
    if not np.isfinite(global_row).all() or not np.isfinite(dogs).all():
        raise ValueError("Nonfinite execution features")
    return {"schema": FEATURE_SCHEMA, "global": global_row, "per_dog": dogs,
            "dog_mask": np.ones(m, dtype=np.float32), "component_mask": components,
            "target_mask": targets, "global_names": GLOBAL_FEATURE_NAMES, "dog_names": DOG_FEATURE_NAMES}


def executor_features(session, commitment=None):
    return executor_features_from_state(session.env, session.co, session.ctx, commitment)


def features(snapshot):
    return snapshot.executor_features if hasattr(snapshot, "executor_features") else executor_features(snapshot.session, snapshot.commitment)

