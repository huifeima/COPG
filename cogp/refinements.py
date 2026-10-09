"""Frozen-backbone proposal refinement and fixed residual ridge heads.

These selectors only change the assignments supplied to COGP. They neither
query the reference simulator at deployment nor update coordination networks.
The exact historic descriptor implementation was not available: FEATURE_NAMES
defines the explicit, deterministic 28-dimensional reconstruction in this
standalone project. Its fitted heads must not be confused with historic heads.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

from .io import atomic_json, digest

FEATURE_NAMES = (
    "native_score", "native_fractional_rank", "travel_mean", "travel_std",
    "travel_max", "travel_min", "travel_upper_quartile", "agent_cost_rank_mean",
    "agent_cost_rank_max", "collect_fraction", "rear_fraction", "left_fraction",
    "right_fraction", "reserve_fraction", "collect_component_coverage",
    "collect_member_coverage", "collect_target_coverage", "collect_target_repetition",
    "agent_best_cost_gap", "task_best_agent_cost_gap", "task_competitor_margin",
    "task_agent_cost_spread", "log_team_size", "log_flock_size", "drive_phase",
    "travel_team_interaction", "collection_flock_interaction", "phase_team_interaction",
)
HEAD_SCHEMA = "fixed-proposal-residual-ridge"
NATIVE_WIDTH, NATIVE_LIMIT, GEOMETRY_LIMIT, SHORTLIST_LIMIT = 4096, 128, 256, 32


@dataclass
class ExpandedSupport:
    specs: list
    native: object
    scores: np.ndarray
    info: dict


def _state(session):
    from .coordination.ranking import prepare
    snapshot = session.snapshot
    if callable(snapshot): snapshot = snapshot()
    return prepare(snapshot)


def _key(spec):
    from .coordination.snapshot import task_key
    return task_key(spec)


def _deduplicate(specs):
    result, seen = [], set()
    for spec in specs:
        key = _key(spec)
        if key not in seen:
            result.append(spec)
            seen.add(key)
    return result


def _native_beam(policy, state):
    """Return the complete final fixed-width native beam in native rank order."""
    import torch
    from .coordination import core
    from .coordination.ranking import Factors, pair_scale_factor, rank_assignments
    m = int(state.structure.active_dogs)
    if m <= 3:
        return rank_assignments(policy, state, state.specs)[0], {"search": "exact"}
    if hasattr(policy, "family_name"):
        from .coordination.external_scores import table_for
        table = table_for(policy, state)
        pool = table.pool
        beam = [()]
        expanded, truncated = 0, False
        for _ in range(m):
            following = []
            for prefix in beam:
                for k in range(len(pool)):
                    if k in prefix: continue
                    q = prefix + (k,)
                    if policy.space.is_prefix_extendable(tuple(pool[j] for j in q), state.structure, state.phase):
                        following.append(q)
            if not following: raise RuntimeError("No feasible native refinement beam")
            values = table.score_indices(following, partial=True).detach().cpu().double().numpy()
            if not np.isfinite(values).all(): raise RuntimeError("Nonfinite native refinement score")
            order = np.argsort(-values, kind="stable")
            expanded += len(following)
            truncated |= len(following) > NATIVE_WIDTH
            beam = [following[int(j)] for j in order[:NATIVE_WIDTH]]
        specs = [core.HCCGAssignmentSpec(tuple(pool[j] for j in q)) for q in beam
                 if policy.space.is_valid(tuple(pool[j] for j in q), state.structure, state.phase)]
    else:
        device = next(policy.parameters()).device
        g = torch.as_tensor(state.global_np, device=device)
        table = Factors(policy, g, state.structure, state.phase, core)
        pool = table.pool
        unary = table.unary.detach().cpu().double().numpy()
        pair = {(i, j): table.pair_block(i, j).detach().cpu().double().numpy()
                for i in range(m) for j in range(i + 1, m)} if policy.pairwise_weight else {}
        multiplier = policy.pairwise_weight * pair_scale_factor(policy, m) * m / math.comb(m, 2)
        beam, expanded, truncated = [(0., ())], 0, False
        for j in range(m):
            following = []
            for value, prefix in beam:
                for k in range(len(pool)):
                    if k in prefix: continue
                    q = prefix + (k,)
                    if not policy.space.is_prefix_extendable(tuple(pool[h] for h in q), state.structure, state.phase): continue
                    v = value + float(unary[j, k])
                    if multiplier:
                        v += sum(multiplier * float(pair[i, j][ki, k]) for i, ki in enumerate(prefix))
                    following.append((v, q))
            if not following: raise RuntimeError("No feasible native refinement beam")
            if not all(math.isfinite(x[0]) for x in following): raise RuntimeError("Nonfinite native factor score")
            following.sort(key=lambda row: row[0], reverse=True)
            expanded += len(following)
            truncated |= len(following) > NATIVE_WIDTH
            beam = following[:NATIVE_WIDTH]
        specs = [core.HCCGAssignmentSpec(tuple(pool[j] for j in q)) for _, q in beam
                 if policy.space.is_valid(tuple(pool[j] for j in q), state.structure, state.phase)]
    if not specs: raise RuntimeError("Native refinement support is empty")
    ranked = rank_assignments(policy, state, specs)[0]
    return ranked, {"search": "native_beam", "width": NATIVE_WIDTH,
                    "expanded": expanded, "truncated": bool(truncated)}


def expanded_support(session):
    """Exact small-team or native-128/geometry-256 large-team support.

    The returned support is the at-most-32 native-logit shortlist. The original
    native winner is retained first even if floating-point near-tie repair in
    its native inference selected a different item than vectorised ranking.
    """
    import torch
    from .coordination.ranking import inference, rank_assignments
    from .coordination.geometry import geometry_candidates
    state = _state(session)
    with torch.no_grad():
        native, native_info = inference(session.policy, state)
        if state.structure.active_dogs <= 3:
            raw = list(state.specs)
            search_info = {"search": "exact", "raw_count": len(raw)}
        else:
            native_beam, search_info = _native_beam(session.policy, state)
            geom, geom_info = geometry_candidates(state, width=GEOMETRY_LIMIT)
            raw = _deduplicate([native] + native_beam[:NATIVE_LIMIT] + geom[:GEOMETRY_LIMIT])
            search_info.update(geometry=geom_info, raw_count=len(raw))
        raw = _deduplicate([native] + raw)
        ordered, scores = rank_assignments(session.policy, state, raw)
        by_key = {_key(spec): float(score) for spec, score in zip(ordered, scores)}
        shortlist = [native] + [s for s in ordered if _key(s) != _key(native)][:SHORTLIST_LIMIT - 1]
        values = np.asarray([by_key[_key(s)] for s in shortlist], np.float64)
        return ExpandedSupport(shortlist, native, values,
                               dict(search_info, shortlist_count=len(shortlist),
                                    native_preserved=True, native_inference=native_info))


def _occupancy(spec):
    from .coordination.core import TASK_COLLECT
    count = Counter()
    for task in spec.tasks_by_dog:
        key = (int(task.task_type), tuple(task.component_members), int(task.target)) if task.task_type == TASK_COLLECT else (int(task.task_type),)
        count[key] += 1 / len(spec.tasks_by_dog)
    return count


def assignment_distance(first, second):
    """0.75 occupancy total variation + 0.25 ordered binding Hamming."""
    if len(first.tasks_by_dog) != len(second.tasks_by_dog) or not first.tasks_by_dog:
        raise ValueError("Diversity requires equal nonempty teams")
    a, b = _occupancy(first), _occupancy(second)
    tv = 0.5 * sum(abs(a[k] - b[k]) for k in set(a) | set(b))
    hamming = np.mean([x.key() != y.key() for x, y in zip(first.tasks_by_dog, second.tasks_by_dog)])
    return float(0.75 * tv + 0.25 * hamming)


def _standard_scores(scores):
    scores = np.asarray(scores, np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Native scores must be a nonempty finite vector")
    scale = float(scores.std())
    return (scores - scores.mean()) / (scale if scale > 1e-8 else 1.)


def features(session, specs, nativescores):
    """Explicit reconstructed fixed-size descriptors, in FEATURE_NAMES order."""
    from .coordination import core
    from .coordination.geometry import geometry_table
    state = _state(session)
    return features_from_state(state, specs, nativescores)


def features_from_state(state, specs, nativescores):
    from .coordination import core
    from .coordination.geometry import geometry_table
    values = np.asarray(nativescores, np.float64)
    if values.shape != (len(specs),) or not len(specs) or not np.isfinite(values).all():
        raise ValueError("Feature candidate/score mismatch")
    pool, costs = geometry_table(state)
    index = {task.key(): k for k, task in enumerate(pool)}
    m, n = int(state.structure.active_dogs), int(state.structure.active_sheep)
    if m < 1 or n < 1: raise ValueError("Empty physical population")
    ranks = np.empty_like(costs)
    for i, row in enumerate(costs):
        order = np.argsort(row, kind="stable")
        ranks[i, order] = np.arange(len(row)) / max(1, len(row) - 1)
    native_order = np.argsort(-values, kind="stable")
    native_rank = np.empty(len(specs), np.float64)
    native_rank[native_order] = np.arange(len(specs)) / max(1, len(specs) - 1)
    available_collect = [t for t in pool if t.task_type == core.TASK_COLLECT]
    available_components = {tuple(t.component_members) for t in available_collect}
    available_members = {i for t in available_collect for i in t.component_members}
    available_targets = {int(t.target) for t in available_collect}
    log_m, log_n, phase = math.log1p(m), math.log1p(n), float(state.phase == core.PHASE_DRIVE)
    output = []
    for si, spec in enumerate(specs):
        if len(spec.tasks_by_dog) != m: raise ValueError("Wrong assignment team size")
        indices = np.asarray([index[t.key()] for t in spec.tasks_by_dog], np.int64)
        travel = costs[np.arange(m), indices]
        fractions = [sum(t.task_type == k for t in spec.tasks_by_dog) / m for k in range(core.NUM_TASK_TYPES)]
        collect = [t for t in spec.tasks_by_dog if t.task_type == core.TASK_COLLECT]
        components = {tuple(t.component_members) for t in collect}
        members = {i for t in collect for i in t.component_members}
        targets = {int(t.target) for t in collect}
        assigned_ranks = ranks[np.arange(m), indices]
        best_agent_gaps = []
        margins, spreads = [], []
        for i, k in enumerate(indices):
            profile = costs[:, k]
            best_agent_gaps.append(float(profile[i] - profile.min()))
            other = np.delete(profile, i)
            margins.append(float(other.min() - profile[i]) if len(other) else 0.)
            spreads.append(float(profile.std()))
        row = [float(values[si]), float(native_rank[si]), float(travel.mean()), float(travel.std()),
               float(travel.max()), float(travel.min()), float(np.quantile(travel, .75)),
               float(assigned_ranks.mean()), float(assigned_ranks.max()), *fractions,
               len(components) / max(1, len(available_components)),
               len(members) / max(1, len(available_members)),
               len(targets) / max(1, len(available_targets)),
               1 - len(targets) / len(collect) if collect else 0.,
               float(np.mean(travel - costs.min(1))), float(np.mean(best_agent_gaps)),
               float(np.mean(margins)), float(np.mean(spreads)), log_m, log_n, phase,
               float(travel.mean()) * log_m, fractions[core.TASK_COLLECT] * log_n, phase * log_m]
        output.append(row)
    result = np.asarray(output, np.float64)
    if result.shape != (len(specs), len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise RuntimeError("Invalid reconstructed proposal descriptor matrix")
    return result


def _load_head(headpath, mode, session=None):
    import json
    if headpath is None or not Path(headpath).is_file():
        raise FileNotFoundError("A fitted proposal head is required; run fit-proposal-head on training reference labels")
    head = json.loads(Path(headpath).read_text())
    if head.get("schema") != HEAD_SCHEMA or head.get("mode") != mode:
        raise ValueError("Wrong proposal head schema or fitting objective")
    if tuple(head.get("feature_names", [])) != FEATURE_NAMES:
        raise ValueError("Proposal head descriptor ordering differs")
    if head.get("reconstructed_descriptor") is not True or head.get("ridge") != 1.:
        raise ValueError("Proposal head is not the documented fixed-ridge reconstruction")
    if session is not None and hasattr(session, "seed") and int(head["seed"]) != int(session.seed):
        raise ValueError("Proposal head training seed differs from frozen backbone")
    if session is not None:
        arm = getattr(session, "arm", {})
        generator = arm.get("generator", getattr(session, "method", None))
        if generator is not None:
            normalize = lambda name: str(name).lower().replace("_", "").replace("-", "")
            if normalize(generator) != normalize(head["generator"]):
                raise ValueError("Proposal head generator differs from frozen backbone")
    for key in ("mean", "scale", "weight"):
        value = np.asarray(head[key], np.float64)
        if value.shape != (len(FEATURE_NAMES),) or not np.isfinite(value).all():
            raise ValueError("Invalid proposal head tensors")
    if np.any(np.asarray(head["scale"]) <= 0): raise ValueError("Invalid proposal head feature scale")
    return head


def residual_scores(head, matrix, native_scores):
    matrix = np.asarray(matrix, np.float64)
    if matrix.shape != (len(native_scores), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError("Wrong proposal feature matrix")
    return _standard_scores(native_scores) + ((matrix - np.asarray(head["mean"])) / np.asarray(head["scale"])) @ np.asarray(head["weight"])


def select_refinement(session, support, nativescores=None, mode="beam", K=4, headpath=None):
    """Return (selected specifications, unchanged native logits, receipt)."""
    k = int(K)
    if k < 1: raise ValueError("Candidate budget must be positive")
    specs = list(support.specs if isinstance(support, ExpandedSupport) else support)
    native = support.native if isinstance(support, ExpandedSupport) else specs[0]
    values = np.asarray(support.scores if nativescores is None and isinstance(support, ExpandedSupport) else nativescores, np.float64)
    if values.shape != (len(specs),) or not len(specs) or not np.isfinite(values).all():
        raise ValueError("Refinement support requires aligned finite native scores")
    native_index = next((i for i, spec in enumerate(specs) if _key(spec) == _key(native)), None)
    if native_index is None: raise ValueError("Original native winner absent from refinement support")
    mode = {"learned": "return_rank", "winner_imitation": "winner"}.get(mode, mode)
    chosen = [native_index]
    remaining = [i for i in range(len(specs)) if i != native_index]
    if mode in ("beam", "native", "original"):
        remaining.sort(key=lambda i: (-float(values[i]), i))
        chosen += remaining[:k - 1]
    elif mode == "diverse":
        while remaining and len(chosen) < k:
            distances = {i: min(assignment_distance(specs[i], specs[j]) for j in chosen) for i in remaining}
            pick = min(remaining, key=lambda i: (-distances[i], -float(values[i]), i))
            chosen.append(pick)
            remaining.remove(pick)
    elif mode in ("return_rank", "winner"):
        head = _load_head(headpath, mode, session)
        matrix = features(session, specs, values)
        refined = residual_scores(head, matrix, values)
        remaining.sort(key=lambda i: (-float(refined[i]), -float(values[i]), i))
        chosen += remaining[:k - 1]
    else:
        raise ValueError("Unknown frozen-backbone proposal refinement: " + str(mode))
    receipt = {"selector": mode, "requested_K": k, "effective_K": len(chosen),
               "support_count": len(specs), "native_preserved": True,
               "chosen_support_indices": chosen, "reference_calls_at_selection": 0}
    return [specs[i] for i in chosen], values[chosen], receipt


def _records(labelrecords):
    import json
    if isinstance(labelrecords, (str, Path)):
        path = Path(labelrecords)
        if path.suffix == ".jsonl":
            result = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            packet = json.loads(path.read_text())
            result = packet.get("states", []) if isinstance(packet, dict) else packet
    else:
        result = list(labelrecords)
    if not result: raise ValueError("Proposal training records are empty")
    return result


def _state_arrays(record):
    candidates = record["candidates"]
    if not 2 <= len(candidates) <= 8:
        raise ValueError("Proposal fitting requires 2--8 labelled candidates per state")
    features_ = np.asarray([row["features"] for row in candidates], np.float64)
    native = np.asarray([row["native_score"] for row in candidates], np.float64)
    reference = np.asarray([row["reference_return"] for row in candidates], np.float64)
    if features_.shape != (len(candidates), len(FEATURE_NAMES)) or not all(np.isfinite(x).all() for x in (features_, native, reference)):
        raise ValueError("Incomplete or nonfinite proposal label records")
    return features_, native, reference


def _pair_rows(record, mode, mean, scale):
    matrix, native, reference = _state_arrays(record)
    z = (matrix - mean) / scale
    native = _standard_scores(native)
    if mode == "return_rank":
        if float(reference.std()) <= 1e-8: return None
        target = (reference - reference.mean()) / reference.std()
    else:
        target = np.zeros(len(reference), np.float64)
        target[int(np.argmax(reference))] = 1.
        target = (target - target.mean()) / target.std()
    x, y = [], []
    for i in range(len(reference)):
        for j in range(i + 1, len(reference)):
            if mode == "return_rank" and reference[i] == reference[j]: continue
            if mode == "winner" and target[i] == target[j]: continue
            x.append(z[i] - z[j])
            y.append(float(target[i] - target[j] - native[i] + native[j]))
    return (np.asarray(x), np.asarray(y)) if x else None


def _diagnostics(records, mean, scale, weight):
    gaps, native_gaps, agreement = [], [], []
    for row in records:
        matrix, native, reference = _state_arrays(row)
        scores = _standard_scores(native) + ((matrix - mean) / scale) @ weight
        selected = int(np.argmax(scores))
        direct = int(np.argmax(native))
        gaps.append(float(reference.max() - reference[selected]))
        native_gaps.append(float(reference.max() - reference[direct]))
        agreement.append(bool(reference[selected] == reference.max()))
    if not records: return {"states": 0}
    return {"states": len(records), "mean_reference_regret": float(np.mean(gaps)),
            "native_mean_reference_regret": float(np.mean(native_gaps)),
            "reference_winner_agreement": float(np.mean(agreement))}


def fit_head(labelrecords, output, mode="return_rank"):
    """Fit one seed/generator head; validation is diagnostic, never a selector.

    Input is JSON/JSONL or state-record dictionaries, with keys state_id, split
    ('train'/'validation'), seed, generator and candidates. Each candidate has
    features[28], native_score and true-simulator reference_return. State names
    must be unique and training/validation world_id groups must be disjoint.
    """
    if mode not in ("return_rank", "winner"): raise ValueError("Unsupported proposal fitting objective")
    records = _records(labelrecords)
    if len({str(row["state_id"]) for row in records}) != len(records): raise ValueError("Duplicate proposal label state")
    seeds = {int(row["seed"]) for row in records}
    generators = {str(row["generator"]) for row in records}
    if len(seeds) != 1 or len(generators) != 1: raise ValueError("Fit one backbone seed and generator per head")
    if any(row.get("split") not in ("train", "validation") for row in records):
        raise ValueError("Evaluation/test states cannot be used to fit or validate proposal heads")
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    if not train: raise ValueError("No training proposal states")
    if any("world_id" not in row for row in records): raise ValueError("World IDs are required to audit split separation")
    if {str(row["world_id"]) for row in train} & {str(row["world_id"]) for row in validation}:
        raise ValueError("Training and validation proposal worlds overlap")
    matrices = [_state_arrays(row)[0] for row in train]
    # Each state's feature distribution carries equal mass, even for differing
    # effective supports. Only training states determine the normalization.
    mean = np.mean([matrix.mean(0) for matrix in matrices], 0)
    second = np.mean([(matrix * matrix).mean(0) for matrix in matrices], 0)
    scale = np.sqrt(np.maximum(second - mean * mean, 0.))
    scale[scale <= 1e-8] = 1.
    gram = np.eye(len(FEATURE_NAMES), dtype=np.float64)
    rhs = np.zeros(len(FEATURE_NAMES), np.float64)
    usable_states, pairs = 0, 0
    for row in train:
        pair = _pair_rows(row, mode, mean, scale)
        if pair is None: continue
        x, y = pair
        gram += x.T @ x / len(x)
        rhs += x.T @ y / len(x)
        usable_states += 1
        pairs += len(x)
    if not usable_states: raise ValueError("No unequal-reference candidate pairs to fit proposal head")
    weight = np.linalg.solve(gram, rhs)
    if not np.isfinite(weight).all(): raise RuntimeError("Nonfinite proposal ridge solution")
    packet = {"schema": HEAD_SCHEMA, "mode": mode, "feature_names": list(FEATURE_NAMES),
              "ridge": 1., "reconstructed_descriptor": True, "seed": next(iter(seeds)),
              "generator": next(iter(generators)), "mean": mean.tolist(), "scale": scale.tolist(),
              "weight": weight.tolist(), "states_train": len(train), "usable_states_train": usable_states,
              "pairs_train": pairs, "training_record_hash": digest(train),
              "training_state_ids": [str(row["state_id"]) for row in train],
              "validation_state_ids": [str(row["state_id"]) for row in validation],
              "validation_used_for_fit": False, "hyperparameter_search": False,
              "train_diagnostic": _diagnostics(train, mean, scale, weight),
              "validation_diagnostic": _diagnostics(validation, mean, scale, weight)}
    atomic_json(output, packet)
    return packet
