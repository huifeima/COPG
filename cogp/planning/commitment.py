"""Resumable isolated commitments for execution-state and horizon controls.

Pausing preserves the same assignment and all loop state. Atomic acquisition
and low-level execution call the shared committed coordinator implementation.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
import numpy as np
from cogp.controls.features import executor_features

def _branching():
    from cogp.planning import branching
    return branching

def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)

def _terminal(env, done=False):
    return bool(done or int(env.primitive_step) >= int(env.cfg.max_steps) or float(env.metrics().get('success', 0.0)) > 0.5)

def _session_copy(template, env, co, ctx):
    result = copy.copy(template)
    result.env, result.co, result.ctx = (env, co, ctx)
    result.tb = co.task_builder
    result.executor = co.executor
    result.controller = co.executor.low_level
    result.pc = co.pc
    result.ec = env.cfg
    result.done = _terminal(env)
    if hasattr(template, 'ob'):
        result.ob = copy.deepcopy(template.ob)
    return result

@dataclass
class Handle:
    session: Any
    spec: Any
    E: int
    predictor: Any
    origin: Any
    origin_digest: str
    start_primitive: int
    previous_limit: int
    initial_ctx: Any
    cur: Any
    start_phase: str
    start_partition: Any
    start_seg: int
    start_abort: int
    exe: Any
    reward: float = 0.0
    raw: float = 0.0
    duration: int = 0
    chunks: int = 0
    discount: float = 1.0
    done: bool = False
    pending: Any = None
    info: dict = field(default_factory=dict)
    collision_any: int = 0
    active: bool = True
    finalized: bool = False
    stop_reason: str = 'active'
    last_request: int = 0

    @property
    def remaining_checks(self):
        return max(0, self.E - self.chunks) if self.active else 0

@dataclass(frozen=True)
class Snapshot:
    session: Any
    commitment: Any
    reward: float
    raw: float
    duration: int
    primitive_steps: int
    done: bool
    active: bool
    remaining_checks: int
    pending: Any
    info: dict
    executor_features: dict
    stop_reason: str

    @property
    def env(self):
        return self.session.env

    @property
    def co(self):
        return self.session.co

    @property
    def ctx(self):
        return self.session.ctx

    @property
    def total_reward(self):
        return self.reward

    @property
    def total_duration(self):
        return self.duration

    @property
    def terminal(self):
        return self.done

    @property
    def boundary(self):
        return not self.active

def start(session, spec, E=64, predictor=None):
    """Clone a live decision state and start one fixed assignment.

    E is the original nominal commitment, never the requested pause location.
    Predictor None means true simulator dynamics (oracle/training use only).
    """
    E = _positive_int(E, 'E')
    if session.ctx is None or _terminal(session.env):
        raise ValueError('A nonterminal prepared decision state is required')
    branching = _branching()
    before = branching.state_digest(session.env, session.co, session.ctx)
    env, co, ctx = branching.clone_branch(session.env, session.co, session.ctx)
    local = _session_copy(session, env, co, ctx)
    previous_limit = int(co.max_option_high_steps)
    co.max_option_high_steps = E
    start_partition = co.task_builder.build_from_env(env).partition_signature
    start_seg = int(co.executor.drive_segments.completed_count)
    start_abort = int(co.acquire_aborted_total + co.acquire_timeout_total)
    exe = co._execution(env, ctx, spec)
    result = Handle(local, spec, E, predictor, session, before, int(env.primitive_step), previous_limit, ctx, ctx, str(ctx.phase), copy.deepcopy(start_partition), start_seg, start_abort, exe)
    _check_origin(result)
    return result

def _check_origin(handle):
    b = _branching()
    origin = handle.origin
    if b.state_digest(origin.env, origin.co, origin.ctx) != handle.origin_digest:
        raise RuntimeError('Live state changed while an isolated commitment was active')

def _finalize(handle):
    if handle.finalized:
        return
    session = handle.session
    env, co = (session.env, session.co)
    handle.info.update({'task_assignment_signature': handle.spec.signature(), 'task_collect_count': float(handle.spec.collect_count), 'task_reserve_count': float(handle.spec.reserve_count), 'task_collision_any': float(handle.collision_any), 'task_option_duration_high': float(handle.duration), 'task_partition_changed': float(co.task_builder.build_from_env(env).partition_signature != handle.start_partition)})
    handle.done = _terminal(env, handle.done)
    session.done = handle.done
    session.ctx = handle.pending if handle.pending is not None else handle.initial_ctx if handle.done else co.prepare(env)
    co.max_option_high_steps = handle.previous_limit
    handle.active = False
    handle.finalized = True

def advance(handle, requested_total_checks):
    """Advance this same commitment to a cumulative check cut or real ending.

    An atomic acquisition chunk can overshoot the requested cut and/or E.
    Returned snapshots own their mutable state and survive later resumption.
    """
    requested = _positive_int(requested_total_checks, 'requested_total_checks')
    if requested < handle.last_request:
        raise ValueError('Pause requests must be nondecreasing; snapshots cannot rewind a handle')
    handle.last_request = requested
    _check_origin(handle)
    if handle.active:
        session = handle.session
        env, co = (session.env, session.co)
        limit = min(requested, handle.E)
        with _branching().dynamics_context(env, handle.predictor):
            while handle.active and handle.chunks < limit and (env.primitive_step < env.cfg.max_steps):
                r, done, info, dur, raw = co._step_custom(env, handle.cur, handle.exe)
                dur = _positive_int(dur, 'atomic duration')
                if not np.isfinite([r, raw]).all():
                    raise RuntimeError('Nonfinite commitment reward')
                handle.reward += handle.discount * float(r)
                handle.discount *= float(co.gamma) ** max(1, dur)
                handle.raw += float(raw)
                handle.duration += dur
                handle.chunks += max(1, dur)
                handle.info = dict(info)
                handle.done = bool(done)
                handle.collision_any = max(handle.collision_any, int(float(info.get('dog_collision_count', 0.0)) > 0), int(float(info.get('collision_primitive_fraction', 0.0)) > 0))
                if done:
                    handle.active, handle.stop_reason = (False, 'episode_terminal')
                    break
                if int(co.acquire_aborted_total + co.acquire_timeout_total) > handle.start_abort and handle.cur.phase == 'DRIVE':
                    st = co.task_builder.build_from_env(env)
                    handle.pending = type(handle.initial_ctx)('DRIVE', 'RECOVER', np.asarray([t.target for t in st.collect_tasks], dtype=np.int64), st.lcc_fraction, True)
                    handle.active, handle.stop_reason = (False, 'acquisition_abort_or_timeout')
                    break
                nxt = co.phase_gate.prepare(env)
                if nxt.transitioned:
                    handle.pending = nxt
                    handle.active, handle.stop_reason = (False, 'phase_transition')
                    break
                if handle.start_phase == 'RECOVER':
                    if co.task_builder.build_from_env(env).partition_signature != handle.start_partition:
                        handle.pending = nxt
                        handle.active, handle.stop_reason = (False, 'partition_change')
                        break
                elif int(co.executor.drive_segments.completed_count) > handle.start_seg:
                    handle.pending = nxt
                    handle.active, handle.stop_reason = (False, 'drive_segment_completion')
                    break
                handle.cur = nxt
            if handle.active and env.primitive_step >= env.cfg.max_steps:
                handle.active, handle.stop_reason = (False, 'episode_time_limit')
            if handle.active and handle.chunks >= handle.E:
                handle.active, handle.stop_reason = (False, 'commitment_budget')
        if not handle.active:
            _finalize(handle)
        else:
            session.ctx = handle.cur
    _check_origin(handle)
    return _snapshot(handle)

def finish(handle):
    """Complete the original assignment, preserving all paused loop state."""
    return advance(handle, max(handle.E, handle.last_request))

def _snapshot(handle):
    local = handle.session
    env, co, ctx = _branching().clone_branch(local.env, local.co, local.ctx)
    result = _session_copy(local, env, co, ctx)
    result.done = bool(handle.done)
    capture = getattr(result, 'capture', None)
    if callable(capture):
        capture()  # Refresh physical/task snapshot only; no successor prepare.
    commitment = SimpleNamespace(spec=copy.deepcopy(handle.spec), exe=copy.deepcopy(handle.exe), E=int(handle.E), elapsed=int(handle.duration), duration=int(handle.duration), active=bool(handle.active), boundary=not handle.active, remaining_checks=handle.remaining_checks, start_phase=handle.start_phase, start_partition=copy.deepcopy(handle.start_partition), start_seg=int(handle.start_seg), start_abort=int(handle.start_abort))
    data = executor_features(result, commitment)
    return Snapshot(result, commitment, float(handle.reward), float(handle.raw), int(handle.duration), int(env.primitive_step) - handle.start_primitive, bool(handle.done), bool(handle.active), handle.remaining_checks, copy.deepcopy(handle.pending), copy.deepcopy(handle.info), data, handle.stop_reason)

def parity_check(session, spec, E=64, predictor=None, cuts=(12,)):
    """Blocking real-session check against the committed coordinator implementation.

    This compares all environment/executor/context memory as well as rewards,
    not just visible positions. Include natural endings and acquisition cases
    in a server gate; one passing case is not universal coverage.
    """
    b = _branching()
    expected = b.rollout_branch(session.env, session.co, session.ctx, spec, predictor, max_high_steps=E)
    handle = start(session, spec, E, predictor)
    snapshots = []
    for cut in sorted(set((int(c) for c in cuts if 0 < int(c) < int(E)))):
        snapshots.append(advance(handle, cut))
    result = finish(handle)
    for name in ('reward', 'raw'):
        if not math.isclose(float(getattr(expected, name)), float(getattr(result, name)), rel_tol=1e-11, abs_tol=1e-11):
            raise AssertionError(f'Commitment parity failed for {name}')
    for name in ('duration', 'primitive_steps', 'done'):
        if getattr(expected, name) != getattr(result, name):
            raise AssertionError(f'Commitment parity failed for {name}')
    # Both paths use E only inside the isolated branch. The resumable handle
    # restores the incoming nominal limit at a genuine end; normalize the
    # comparison wrapper's temporary override before comparing mutable state.
    expected.co.max_option_high_steps = int(session.co.max_option_high_steps)
    expected_digest = b.state_digest(expected.env, expected.co, expected.ctx)
    actual_digest = b.state_digest(result.env, result.co, result.ctx)
    if actual_digest != expected_digest:
        raise AssertionError('Commitment parity failed for full mutable state')
    return {'passed': True, 'reward': result.reward, 'duration': result.duration, 'primitive_steps': result.primitive_steps, 'stop_reason': result.stop_reason, 'state_digest': actual_digest, 'cuts': [{'duration': s.duration, 'active': s.active, 'remaining_checks': s.remaining_checks} for s in snapshots], 'dynamics': 'true_simulator' if predictor is None else 'learned'}

def gate_session(session, model, specs=None, E=64):
    """Checkpoint-backed blocking gate for one prepared server decision state.

    The caller supplies the native proposal and any tested alternatives.
    The runner should call this at multiple sizes/decision states. Internal
    H12 pauses match the short fixed-check forecast's physical inputs and score
    components; their gate memory intentionally omits the artificial
    successor prepare call. Complete E64 states must match exactly.
    """
    b = _branching()
    before = b.state_digest(session.env, session.co, session.ctx)
    if specs is None:
        raise ValueError('Provide the frozen ordered feasible candidates explicitly')
    if not specs:
        raise ValueError('Parity gate requires at least one feasible candidate')
    records = []
    for index, spec in enumerate(specs):
        for name, predictor in (('true', None), ('learned', model)):
            if name == 'learned' and model is None:
                continue
            result = parity_check(session, spec, E=E, predictor=predictor, cuts=(12,))
            expected = b.rollout_branch(session.env, session.co, session.ctx, spec, predictor, max_high_steps=min(12, int(E)))
            handle = start(session, spec, E=E, predictor=predictor)
            paused = advance(handle, min(12, int(E)))
            for field_name in ('reward', 'raw'):
                if not math.isclose(float(getattr(expected, field_name)), float(getattr(paused, field_name)), rel_tol=1e-11, abs_tol=1e-11):
                    raise AssertionError(f'Short-cut parity failed for {field_name}')
            if (expected.duration, expected.primitive_steps, expected.done) != (paused.duration, paused.primitive_steps, paused.done):
                raise AssertionError('Short-cut duration/termination parity failed')
            physical_a = session.ob.global_state_vector(expected.env)
            physical_b = session.ob.global_state_vector(paused.env)
            if not np.array_equal(physical_a, physical_b):
                raise AssertionError('Short-cut physical critic input differs')
            records.append({'candidate': index, 'source': name, 'full': result, 'short_physical_score_components_match': True, 'short_duration': paused.duration, 'short_active': paused.active})
    if b.state_digest(session.env, session.co, session.ctx) != before:
        raise AssertionError('Parity gate mutated the source session')
    return {'passed': True, 'checks': records, 'live_state_unchanged': True, 'short_memory_note': 'Internal pauses omit successor prepare; only genuine endpoints match full memory.', 'coverage': 'these supplied states/candidates only; runner supplies multiple size/event cases'}
