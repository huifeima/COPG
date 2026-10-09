"""True-simulator labels for frozen proposal-refinement ridge heads.

All generators are compared at the same HCCG+COGP-driven states. Shared joint
assignments are simulated only once at each state. The deployed selectors do
not call this collector and never receive the true reference return.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import time

import numpy as np

from cogp.io import atomic_json, digest, sha
from cogp.refinements import expanded_support, features

SCENARIOS = ("compact", "single_outlier", "double_outlier", "split")
GENERATORS = ("hccg", "edge_gnn", "agent_dag")
SCHEMA = "proposal-reference-training-labels"
DECISIONS, CANDIDATE_LABEL_LIMIT = (0, 2), 8
BASE_ARM = dict(generator="hccg", planner=True, dynamics="graph", critic="original",
                H=64, E=64, K=4, prior=0., actor="default", memory="inherit", proposal="original")


def _read_manifest(path, split, expected):
    if path is None: raise FileNotFoundError(f"An explicit {split} proposal-label world manifest is required")
    path = Path(path).expanduser().resolve()
    content = json.loads(path.read_text())
    if not isinstance(content, list) or len(content) != expected:
        raise ValueError(f"Proposal fitting requires exactly {expected} {split} worlds")
    ids, physical, counts = set(), set(), Counter()
    for case in content:
        ident = case.get("case_id", "")
        if not isinstance(ident, str) or not ident or any(x in ident for x in ("/", "\\", "..")) or ident in ids:
            raise ValueError("Unsafe or duplicated proposal-label case ID")
        if case.get("kind") != "canonical" or case.get("sheep") != 15 or case.get("dogs") != 2:
            raise ValueError("Proposal heads fit only canonical N15 M2 worlds")
        if case.get("scenario") not in SCENARIOS or type(case.get("seed")) is not int or case["seed"] < 0:
            raise ValueError("Invalid proposal-label world reset")
        if float(case.get("physics_multiplier", 1.)) != 1.:
            raise ValueError("Proposal label fitting does not use physics-stress worlds")
        key = (case["seed"], case["scenario"], case["sheep"], case["dogs"], case["kind"])
        if key in physical: raise ValueError("Duplicated physical proposal-label world")
        ids.add(ident); physical.add(key); counts[case["scenario"]] += 1
    if any(counts[scenario] != expected // len(SCENARIOS) for scenario in SCENARIOS):
        raise ValueError("Proposal label worlds must equally cover all four canonical scenarios")
    return copy.deepcopy(content), path, physical


def _state_hash(session):
    from cogp.planning.branching import state_digest
    return state_digest(session.env, session.co, session.ctx)


def collect_world(driver, templates, case, split, seed, *, planner=None,
                  candidate_builder=expanded_support, feature_builder=features,
                  reference_fn=None, state_session_fn=None, state_hash_fn=None):
    """Collect one world at the declared boundaries; explicit test hooks only.

    `templates` maps the three generator names to frozen Session instances.
    Their live execution fields are cloned from the driver before each state.
    Shared candidates have one true-first-option plus greedy-HCCG reference.
    """
    from cogp.coordination.snapshot import task_key
    if split not in ("train", "validation"):
        raise ValueError("Proposal labels require an explicit training/validation split")
    if set(templates) != set(GENERATORS): raise ValueError("All three proposal generator templates are required")
    if reference_fn is None:
        from cogp.diagnostics import reference
        reference_fn = reference
    if state_session_fn is None:
        from cogp.diagnostics import _state_session
        state_session_fn = _state_session
    if state_hash_fn is None: state_hash_fn = _state_hash
    if planner is None:
        from cogp.planning.planner import Planner
        planner = Planner(driver, H=64, E=64, K=4, prior=0., memory="inherit", proposal="original")
    driver.reset(case)
    records, missing, states, unique_references, reused_references = [], [], 0, 0, 0
    first_segment_primitives = 0
    begun = time.perf_counter()
    for decision in range(max(DECISIONS) + 1):
        if driver.done:
            missing.extend(index for index in DECISIONS if index >= decision)
            break
        if decision in DECISIONS:
            states += 1
            state_before = state_hash_fn(driver)
            cache = {}
            for generator in GENERATORS:
                local = state_session_fn(templates[generator], driver)
                support = candidate_builder(local)
                matrix = np.asarray(feature_builder(local, support.specs, support.scores), np.float64)
                if matrix.shape != (len(support.specs), 28) or not np.isfinite(matrix).all():
                    raise RuntimeError("Invalid proposal descriptor matrix during label collection")
                candidates = []
                for index, (spec, score, vector) in enumerate(zip(support.specs, support.scores, matrix)):
                    if index >= CANDIDATE_LABEL_LIMIT: break
                    key = task_key(spec)
                    if key not in cache:
                        result = reference_fn(local, spec, driver, kind="hccg_direct", E=64)
                        if isinstance(result, tuple):
                            reference, first = result
                            first_segment_primitives += int(first.primitive_steps)
                        else:
                            reference = result
                        value = float(reference["reference_return"])
                        if not np.isfinite(value): raise RuntimeError("Nonfinite true proposal reference label")
                        cache[key] = copy.deepcopy(reference)
                        unique_references += 1
                    else:
                        reused_references += 1
                    reference = cache[key]
                    candidates.append(dict(candidate_key=key, support_index=index,
                        features=vector.tolist(), native_score=float(score),
                        reference_return=float(reference["reference_return"]),
                        reference=copy.deepcopy(reference)))
                    if state_hash_fn(driver) != state_before:
                        raise RuntimeError("Proposal reference branch modified its common live driver")
                # Very small supports cannot define a pairwise ranking target;
                # record the omission instead of inventing alternatives.
                if len(candidates) < 2:
                    missing.append(dict(decision=decision, generator=generator, reason="fewer-than-two-feasible-candidates"))
                    continue
                records.append(dict(state_id=f"s{seed}/{case['case_id']}/decision{decision}/{generator}",
                    world_id=case["case_id"], seed=int(seed), generator=generator, split=split,
                    decision=decision, driver="hccg_cogp", candidates=candidates,
                    support_info=copy.deepcopy(support.info), full_shortlist_count=len(support.specs),
                    source_state_hash=state_before, reference_continuation="frozen-greedy-hccg-to-terminal"))
            if state_hash_fn(driver) != state_before:
                raise RuntimeError("Proposal collection changed the sampling state")
        if decision < max(DECISIONS):
            selection = planner.select()
            driver.step(selection.spec)
    return records, dict(case_id=case["case_id"], split=split, states=states,
        missing_decisions=missing, unique_reference_rollouts=unique_references,
        reused_reference_labels=reused_references, first_segment_primitive_steps=first_segment_primitives,
        collection_seconds=time.perf_counter() - begun)


def collect(registry, seed, train_manifest, validation_manifest, out, device="cpu"):
    """Collect fixed 16/8 canonical worlds and return per-state fitting records.

    Manifests are required because historic proposal training resets were not
    supplied. The caller must use archived reset manifests or explicitly label
    its supplied manifests as a new replication. Existing compatible outputs
    are reused only after their full identity and record digest match.
    """
    from cogp.assets import AssetRegistry
    from cogp.runtime import Session
    seed = int(seed)
    if seed not in range(201, 209): raise ValueError("Proposal-label seed is outside the paper's frozen seed panel")
    registry = registry if isinstance(registry, AssetRegistry) else AssetRegistry(registry)
    training, train_path, train_keys = _read_manifest(train_manifest, "train", 16)
    validation, validation_path, validation_keys = _read_manifest(validation_manifest, "validation", 8)
    if train_keys & validation_keys or {w["case_id"] for w in training} & {w["case_id"] for w in validation}:
        raise ValueError("Training and validation proposal worlds overlap")
    out = Path(out).expanduser().resolve()
    if out.suffix.lower() != ".json": out = out / f"proposal_labels_seed_{seed}.json"
    driver = Session.from_registry(registry, seed, BASE_ARM, device)
    templates = {generator: Session.from_registry(registry, seed, {**BASE_ARM, "generator": generator}, device)
                 for generator in GENERATORS}
    for session in [driver, *templates.values()]:
        session.policy.eval()
        for parameter in session.policy.parameters(): parameter.requires_grad_(False)
    identity = dict(schema=SCHEMA, seed=seed, driver="frozen-hccg-cogp", decision_indices=list(DECISIONS),
        candidate_labels_limit=CANDIDATE_LABEL_LIMIT, H=64, E=64, K=4, prior=0.,
        training_manifest_sha256=sha(train_path), validation_manifest_sha256=sha(validation_path),
        assets={generator: session.asset_identity for generator, session in templates.items()},
        driver_assets=driver.asset_identity, descriptor_reconstruction=True,
        collector_source_sha256=sha(__file__),
        descriptors_source_sha256=sha(Path(__file__).resolve().parents[1] / "refinements.py"),
        reference="true-first-commitment-then-frozen-greedy-hccg-to-terminal")
    expected_order = [(split, case["case_id"]) for split, worlds in (("train", training), ("validation", validation)) for case in worlds]
    if out.exists():
        existing = json.loads(out.read_text())
        if existing.get("identity") != identity or existing.get("record_digest") != digest(existing.get("states", [])):
            raise RuntimeError("Existing proposal-label output identity or checksum differs")
        receipt_order = [(receipt["split"], receipt["case_id"]) for receipt in existing.get("receipts", [])]
        if existing.get("complete") is not True or receipt_order != expected_order:
            raise RuntimeError("Existing proposal label file is not a complete declared world panel")
        if any((row["split"], row["world_id"]) not in set(expected_order) for row in existing.get("states", [])):
            raise RuntimeError("Existing proposal states lie outside the declared training panel")
        return existing["states"]
    states, receipts = [], []
    partial = out.with_suffix(".partial.json")
    if partial.exists():
        saved = json.loads(partial.read_text())
        if saved.get("identity") != identity or saved.get("record_digest") != digest(saved.get("states", [])):
            raise RuntimeError("Partial proposal labels have a different identity or record checksum")
        saved_receipts = saved.get("receipts", [])
        saved_order = [(receipt["split"], receipt["case_id"]) for receipt in saved_receipts]
        if saved_order != expected_order[:len(saved_order)] or saved.get("completed_worlds") != len(saved_order):
            raise RuntimeError("Partial proposal labels do not form the declared world-order prefix")
        if any((row["split"], row["world_id"]) not in set(saved_order) for row in saved.get("states", [])):
            raise RuntimeError("Partial proposal label states lie outside their verified world receipts")
        states, receipts = saved["states"], saved_receipts
    already_completed = {(receipt["split"], receipt["case_id"]) for receipt in receipts}
    for split, worlds in (("train", training), ("validation", validation)):
        for case in worlds:
            if (split, case["case_id"]) in already_completed: continue
            records, receipt = collect_world(driver, templates, case, split, seed)
            states.extend(records); receipts.append(receipt)
            # Checkpoint complete worlds so an interruption preserves verified
            # work. Its path is explicitly a partial output, never a final set.
            atomic_json(out.with_suffix(".partial.json"), dict(schema=SCHEMA, identity=identity,
                completed_worlds=len(receipts), expected_worlds=24, states=states,
                receipts=receipts, record_digest=digest(states), complete=False))
    if not states: raise RuntimeError("No proposal training states were collected")
    packet = dict(schema=SCHEMA, identity=identity, states=states, receipts=receipts,
        record_digest=digest(states), complete=True,
        expected_states_before_early_termination=24 * len(DECISIONS) * len(GENERATORS),
        note="new explicit descriptor reconstruction; not historical fitted-head replay")
    atomic_json(out, packet)
    if partial.exists(): partial.unlink()
    return states
