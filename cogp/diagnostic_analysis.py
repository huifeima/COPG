"""Seed-first descriptive analysis of finite-candidate simulator diagnostics.

Reference returns depend on the explicitly recorded continuation policy.
True-first-segment scores retain the approximate critic and are not oracle
values. Candidate-union coverage is never coverage of the feasible space.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import csv
import json
import math
from pathlib import Path
import re
import statistics

from .io import atomic_json, digest, sha

CROSS_GENERATORS = ("hccg", "edge_gnn", "agent_dag", "geometry")
HISTORY_FIELDS = (
    "_stage_active", "_stage_armed", "_stage_target", "_stage_hold",
    "_prev_high_collect_count", "_prev_high_roles", "_prev_high_targets",
    "_rear_barrier_active", "_rear_barrier_hold", "_rear_barrier_start_primitive",
)
METRICS = (
    "best_of_K", "union_coverage", "within_set_regret", "true_first_regret",
    "best_candidate_agreement", "true_first_best_agreement", "selected_gain",
    "true_first_selected_gain", "true_dynamics_selection_gain", "full_score_discrepancy_mae",
    "first_reward_discrepancy_mae", "effective_K", "union_candidates",
)


class DiagnosticError(ValueError):
    pass


def _finite(value, name):
    try: value = float(value)
    except (TypeError, ValueError) as error: raise DiagnosticError(f"Invalid {name}") from error
    if not math.isfinite(value): raise DiagnosticError(f"Nonfinite {name}")
    return value


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)


def _generator(value):
    key = str(value).lower().replace("_", "").replace("-", "")
    aliases = {"hccg": "hccg", "edgegnn": "edge_gnn", "agentdag": "agent_dag", "geometry": "geometry"}
    if key not in aliases: raise DiagnosticError("Unknown diagnostic generator: " + str(value))
    return aliases[key]


def _identity(tasks):
    if not isinstance(tasks, (tuple, list)) or not tasks:
        raise DiagnosticError("Each candidate needs its ordered task identity")
    return json.dumps(tasks, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_states(path):
    """Read full state records, preserving receipt arm and panel metadata."""
    path = Path(path).expanduser().resolve()
    if not path.exists(): raise FileNotFoundError(path)
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    states, inputs = [], []
    for file in files:
        if file.name.endswith((".sha256.json", ".error.json", ".partial.json")): continue
        packet = json.loads(file.read_text())
        if isinstance(packet, dict) and "status" in packet and "result" in packet:
            if packet["status"] != "COMPLETE": raise DiagnosticError("Input receipt is incomplete: " + str(file))
            sidecar = file.with_suffix(file.suffix + ".sha256.json")
            if sidecar.exists() and json.loads(sidecar.read_text()).get("sha256") != sha(file):
                raise DiagnosticError("Diagnostic receipt checksum mismatch: " + str(file))
        result = packet.get("result", packet) if isinstance(packet, dict) else packet
        rows = result.get("states", []) if isinstance(result, dict) else result if isinstance(result, list) else []
        if not rows: continue
        if any(not isinstance(row, dict) or "state_id" not in row for row in rows):
            raise DiagnosticError("Input has incompatible state records: " + str(file))
        job = packet.get("job", {}) if isinstance(packet, dict) else {}
        for row in rows:
            local = copy.deepcopy(row)
            if job.get("arm_config") is not None: local["_arm_config"] = job["arm_config"]
            elif isinstance(job.get("cohort_arms"), dict) and local.get("generator") in job["cohort_arms"]:
                local["_arm_config"] = job["cohort_arms"][local["generator"]]
            local.setdefault("arm", job.get("arm", result.get("arm", "") if isinstance(result, dict) else ""))
            local.setdefault("panel", job.get("panel", ""))
            local["_source_file"] = str(file)
            states.append(local)
        inputs.append({"path": str(file), "sha256": sha(file), "states": len(rows)})
    if not states: raise DiagnosticError("No full diagnostic state records; summary means cannot reconstruct candidate metrics")
    return states, inputs


def _cohort(state):
    configuration = state.get("_arm_config", state.get("arm_config"))
    if isinstance(configuration, dict):
        fixed = {key: value for key, value in configuration.items() if key != "generator"}
        return digest(fixed), fixed
    panel, arm = str(state.get("panel", "")), str(state.get("arm", ""))
    if panel == "cross_generator" or arm in CROSS_GENERATORS or not arm:
        return "shared-configuration", {"source": "state-record-default"}
    return arm, {"arm": arm, "source": "arm-record"}


def _candidate_arrays(state):
    candidates = state.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise DiagnosticError("State lacks full per-candidate records: " + str(state.get("state_id")))
    identities = [_identity(row.get("tasks")) for row in candidates]
    if len(set(identities)) != len(identities): raise DiagnosticError("Duplicate candidate identities within a generator/state")
    scores = [_finite(row.get("score"), "full candidate score") for row in candidates]
    true_scores = [_finite(row.get("true_first_score"), "true-first-segment score") for row in candidates]
    continuations = set(candidates[0].get("references", {}))
    if not continuations: raise DiagnosticError("Candidate continuation reference returns are missing")
    for row in candidates:
        if set(row.get("references", {})) != continuations:
            raise DiagnosticError("Candidates use inconsistent continuation references")
    references = {kind: [_finite(row["references"][kind].get("reference_return"), "continuation reference return") for row in candidates]
                  for kind in sorted(continuations)}
    discrepancy = []
    reward_error = []
    force_error = []
    force_sources = []
    force_units = []
    for row, score, true_score in zip(candidates, scores, true_scores):
        actual = score - true_score
        if "full_score_discrepancy" in row and not _close(_finite(row["full_score_discrepancy"], "stored score discrepancy"), actual):
            raise DiagnosticError("Stored full-score discrepancy does not equal full score minus true-first score")
        discrepancy.append(abs(actual))
        if "first_reward_discrepancy" in row:
            reward_error.append(abs(_finite(row["first_reward_discrepancy"], "first reward discrepancy")))
        for field in ("force_mae", "force_error", "acceleration_mae", "force_prediction_error"):
            if field in row:
                value = row[field]
                source, unit = field, None
                if isinstance(value, dict):
                    if field != "force_error": raise DiagnosticError("Only force_error supports the verified RMSE diagnostic packet")
                    source, unit = "force_error.rmse", value.get("rmse_unit")
                    if unit != "physical_acceleration_coordinate":
                        raise DiagnosticError("Force RMSE packet does not declare physical acceleration-coordinate units")
                    count, calls = value.get("count"), value.get("force_calls")
                    if type(count) is not int or count < 0 or type(calls) is not int or calls < 0:
                        raise DiagnosticError("Invalid force RMSE sample/force-call counts")
                    if value.get("coordinate_count") != count:
                        raise DiagnosticError("Force RMSE count and coordinate_count differ")
                    squared = _finite(value.get("total_squared_error"), "force total squared error")
                    if squared < 0: raise DiagnosticError("Negative force total squared error")
                    if count == 0:
                        if value.get("rmse") is not None or squared != 0. or calls != 0:
                            raise DiagnosticError("Empty force RMSE packet has nonempty measurements")
                        force_error.append(None)
                    else:
                        rmse = _finite(value.get("rmse"), "recorded physical acceleration RMSE")
                        if rmse < 0 or calls < 1 or not _close(rmse, math.sqrt(squared / count)):
                            raise DiagnosticError("Force RMSE is inconsistent with verified squared errors and counts")
                        if "sheep_accelerations" in value and value["sheep_accelerations"] * 2 != count:
                            raise DiagnosticError("Force acceleration count does not match two coordinate samples per sheep")
                        force_error.append(rmse)
                elif isinstance(value, list):
                    # An explicitly stored vector error is converted to its
                    # elementwise absolute mean, not an unstored physical norm.
                    flat = []
                    def flatten(x):
                        if isinstance(x, list):
                            for child in x: flatten(child)
                        else: flat.append(abs(_finite(x, field)))
                    flatten(value)
                    if not flat: raise DiagnosticError("Empty recorded force errors")
                    force_error.append(statistics.mean(flat))
                else:
                    value = _finite(value, field)
                    if field.endswith("mae") and value < 0: raise DiagnosticError("Negative recorded force MAE")
                    force_error.append(abs(value))
                force_sources.append(source)
                force_units.append(unit)
                break
    if reward_error and len(reward_error) != len(candidates):
        raise DiagnosticError("Only part of a candidate set has first-reward discrepancies")
    if force_error and len(force_error) != len(candidates):
        raise DiagnosticError("Only part of a candidate set has recorded force errors")
    if len(set(force_sources)) > 1:
        raise DiagnosticError("Candidate force-error fields have different definitions")
    if len(set(force_units)) > 1:
        raise DiagnosticError("Candidate force errors have different recorded units")
    measured_force = [value for value in force_error if value is not None]
    if measured_force and len(measured_force) != len(force_error):
        raise DiagnosticError("Only part of a candidate set has observed force samples")
    return dict(candidates=candidates, identities=identities, scores=scores, true_scores=true_scores,
                references=references, discrepancy=statistics.mean(discrepancy),
                reward_error=statistics.mean(reward_error) if reward_error else None,
                force_error=statistics.mean(measured_force) if measured_force else None,
                force_error_source=force_sources[0] if force_sources else None,
                force_error_unit=force_units[0] if force_units else None)


def _native_index(state, size):
    """Use explicit native provenance; ordering alone does not establish it."""
    if state.get("native_verified") is True:
        index = state.get("native_index")
        if type(index) is not int or not 0 <= index < size:
            raise DiagnosticError("Verified native candidate index is invalid")
        return index
    info = state.get("candidate_info", {})
    if "native_index" in info:
        index = info["native_index"]
        if type(index) is not int or not 0 <= index < size:
            raise DiagnosticError("Declared native candidate index is invalid")
        return index
    return None


def _pick(values):
    return max(range(len(values)), key=lambda i: values[i])  # earliest wins exact ties


def analyze_states(states):
    """Return matched finite-set metrics with equal-state then equal-seed means."""
    if not states: raise DiagnosticError("No diagnostic states")
    prepared, groups, cohort_defs = [], defaultdict(list), {}
    cohort_generators = defaultdict(set)
    for state in states:
        if "selected_changed" in state and "candidates" not in state:
            raise DiagnosticError("Use summarize_memory for history-intervention records")
        generator = _generator(state.get("generator"))
        cohort, configuration = _cohort(state)
        cohort_defs[cohort] = configuration
        seed = int(state["seed"])
        label = (str(state.get("panel", "")), cohort, str(state.get("regime", "native_hybrid")), seed, str(state["state_id"]))
        cohort_generators[label[:3]].add(generator)
        item = dict(state=state, generator=generator, seed=seed, cohort=cohort, data=_candidate_arrays(state), group=label)
        prepared.append(item); groups[label].append(item)
    state_rows = []
    for label, items in sorted(groups.items()):
        panel, cohort, regime, seed, state_id = label
        generators = {item["generator"] for item in items}
        if len(generators) != len(items): raise DiagnosticError("Duplicate generator/state records in the same configuration")
        cross = panel == "cross_generator" or len(cohort_generators[label[:3]]) > 1
        if cross and generators != set(CROSS_GENERATORS):
            raise DiagnosticError(f"Incomplete cross-generator set at {state_id}/{regime}: got {sorted(generators)}; need {list(CROSS_GENERATORS)}")
        if cross and len({len(item["data"]["candidates"]) for item in items}) != 1:
            raise DiagnosticError("Cross-generator effective candidate counts are not matched at " + state_id)
        stamps = [item["state"].get("source_state_hash") for item in items]
        if cross and any(stamp is not None for stamp in stamps):
            if any(stamp is None for stamp in stamps) or len(set(stamps)) != 1:
                raise DiagnosticError("Cross-generator source-state hashes are not matched at " + state_id)
        kinds = set(items[0]["data"]["references"])
        if any(set(item["data"]["references"]) != kinds for item in items):
            raise DiagnosticError("Cross-generator records use different continuation targets")
        for kind in sorted(kinds):
            union = {}
            for item in items:
                for identity, reference in zip(item["data"]["identities"], item["data"]["references"][kind]):
                    if identity in union and not _close(union[identity], reference):
                        raise DiagnosticError("Identical joint candidate has inconsistent reference returns in common state")
                    union.setdefault(identity, reference)
            union_best = max(union.values())
            union_winners = {identity for identity, value in union.items() if _close(value, union_best)}
            for item in items:
                data = item["data"]; reference = data["references"][kind]
                selected, true_selected = _pick(data["scores"]), _pick(data["true_scores"])
                best = max(reference)
                native = _native_index(item["state"], len(reference))
                state_rows.append(dict(panel=panel, cohort=cohort, regime=regime, seed=seed, state_id=state_id,
                    generator=item["generator"], continuation=kind, best_of_K=best,
                    union_coverage=int(bool(set(data["identities"]) & union_winners)) if cross else None,
                    within_set_regret=best - reference[selected], true_first_regret=best - reference[true_selected],
                    best_candidate_agreement=int(_close(reference[selected], best)),
                    true_first_best_agreement=int(_close(reference[true_selected], best)),
                    selected_gain=reference[selected] - reference[native] if native is not None else None,
                    true_first_selected_gain=reference[true_selected] - reference[native] if native is not None else None,
                    true_dynamics_selection_gain=reference[true_selected] - reference[selected],
                    full_score_discrepancy_mae=data["discrepancy"], first_reward_discrepancy_mae=data["reward_error"],
                    force_error=data["force_error"], effective_K=len(reference), union_candidates=len(union) if cross else None,
                    force_error_source=data["force_error_source"],
                    force_error_unit=data["force_error_unit"],
                    selected_index=selected, true_first_selected_index=true_selected, native_index=native,
                    native_verified=native is not None))
    seed_groups = defaultdict(list)
    for row in state_rows:
        key = tuple(row[field] for field in ("panel", "cohort", "regime", "generator", "continuation", "seed"))
        seed_groups[key].append(row)
    seed_rows = []
    for key, rows in sorted(seed_groups.items()):
        result = dict(zip(("panel", "cohort", "regime", "generator", "continuation", "seed"), key))
        result["states"] = len(rows)
        force_sources = {row["force_error_source"] for row in rows if row["force_error_source"] is not None}
        if len(force_sources) > 1: raise DiagnosticError("State force-error definitions differ within a seed summary")
        result["force_error_source"] = next(iter(force_sources)) if force_sources else None
        force_units = {row["force_error_unit"] for row in rows if row["force_error_unit"] is not None}
        if len(force_units) > 1: raise DiagnosticError("State force-error units differ within a seed summary")
        result["force_error_unit"] = next(iter(force_units)) if force_units else None
        for metric in (*METRICS, "force_error"):
            values = [row[metric] for row in rows if row.get(metric) is not None]
            result[metric] = statistics.mean(values) if values else None
            result[metric + "_states"] = len(values)
        seed_rows.append(result)
    summary_groups = defaultdict(list)
    for row in seed_rows:
        key = tuple(row[field] for field in ("panel", "cohort", "regime", "generator", "continuation"))
        summary_groups[key].append(row)
    summary = []
    for key, rows in sorted(summary_groups.items()):
        result = dict(zip(("panel", "cohort", "regime", "generator", "continuation"), key))
        result.update(training_seeds=[row["seed"] for row in rows], training_seed_count=len(rows),
                      nested_states=sum(row["states"] for row in rows), state_counts_by_seed={str(row["seed"]): row["states"] for row in rows})
        force_sources = {row["force_error_source"] for row in rows if row["force_error_source"] is not None}
        if len(force_sources) > 1: raise DiagnosticError("Training-seed force-error definitions differ")
        result["force_error_source"] = next(iter(force_sources)) if force_sources else None
        force_units = {row["force_error_unit"] for row in rows if row["force_error_unit"] is not None}
        if len(force_units) > 1: raise DiagnosticError("Training-seed force-error units differ")
        result["force_error_unit"] = next(iter(force_units)) if force_units else None
        for metric in (*METRICS, "force_error"):
            values = [row[metric] for row in rows if row.get(metric) is not None]
            result[metric + "_mean"] = statistics.mean(values) if values else None
            result[metric + "_seed_sd"] = statistics.stdev(values) if len(values) > 1 else None
            result[metric + "_seed_count"] = len(values)
            result[metric + "_states"] = sum(row[metric + "_states"] for row in rows)
        summary.append(result)
    return dict(schema="finite-candidate-descriptive-diagnostics", summary=summary, seed_results=seed_rows,
        state_results=state_rows, configurations=cohort_defs, inferential_intervals=False,
        statistical_unit="equal-state means within training seed, then equal training-seed means",
        scope="fixed sampled states and specified continuation; finite deduplicated union, not optimal action values",
        true_first_scope="true first-segment rollout retains the approximate terminal critic; not an oracle or global upper bound",
        gain_baseline="selected_gain and true_first_selected_gain use the explicitly verified/declared native candidate in the analyzed pool; unknown native provenance yields null",
        true_dynamics_selection_gain_definition="reference return selected by true-first score minus reference return selected by learned score; equals learned regret minus true-first regret",
        force_error_aggregation="equal candidate RMSE means within state, then equal state means within seed, then equal seed means; not a pooled force-sample RMSE",
        discrepancy_definition="absolute(full candidate score minus true-first-segment full score), including both reward and terminal terms")


def _csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def analyze(path, out, plot=False):
    states, inputs = read_states(path)
    result = analyze_states(states)
    result["inputs"] = inputs
    out = Path(out).expanduser().resolve()
    target = out if out.suffix == ".json" else out / "diagnostics_summary.json"
    folder = target.parent
    atomic_json(target, result)
    _csv(folder / "diagnostics_by_seed.csv", result["seed_results"])
    _csv(folder / "diagnostics_by_state.csv", result["state_results"])
    _csv(folder / "diagnostics_summary.csv", result["summary"])
    if plot: result["figures"] = plot_summary(result, folder / "figures")
    if plot: atomic_json(target, result)
    return result


def plot_summary(result, out):
    """Publication-export plots; error bars are seed SD, never state-level CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for row in result["summary"]:
        groups[(row["panel"], row["cohort"], row["regime"], row["continuation"])].append(row)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    figures = []
    labels = {"hccg": "HCCG", "edge_gnn": "Edge-GNN", "agent_dag": "Agent-DAG", "geometry": "Geometry"}
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: CROSS_GENERATORS.index(row["generator"]))
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)
        positions = list(range(len(rows)))
        axes[0].bar(positions, [row["best_of_K_mean"] for row in rows],
                    yerr=[row["best_of_K_seed_sd"] or 0. for row in rows], color="#467c9c", capsize=3)
        axes[0].set_ylabel("Reference best-of-K return")
        width = .36
        axes[1].bar([x - width / 2 for x in positions], [row["within_set_regret_mean"] for row in rows], width,
                    yerr=[row["within_set_regret_seed_sd"] or 0. for row in rows], color="#467c9c", capsize=3, label="Learned first segment")
        axes[1].bar([x + width / 2 for x in positions], [row["true_first_regret_mean"] for row in rows], width,
                    yerr=[row["true_first_regret_seed_sd"] or 0. for row in rows], color="#83a87b", capsize=3, label="True first segment")
        axes[1].set_ylabel("Within-set reference regret")
        axes[1].legend(frameon=False, fontsize=7)
        for axis in axes:
            axis.set_xticks(positions, [labels[row["generator"]] for row in rows], fontsize=8)
            axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{key[2]}; {key[3]} continuation; error bars: training-seed SD", fontsize=8)
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", "_".join(key))[:150]
        for suffix in ("pdf", "png"):
            file = out / (stem + "." + suffix)
            fig.savefig(file, dpi=200)
            figures.append(str(file))
        plt.close(fig)
    return figures


def summarize_memory(states):
    """Sparse shared-state memory diagnostics, separate from closed-loop success."""
    if not states: raise DiagnosticError("No memory-intervention states")
    seen, groups, details = set(), defaultdict(list), []
    field_groups = {"previous_allocation": tuple(k for k in HISTORY_FIELDS if k.startswith("_prev_")),
                    "staging": tuple(k for k in HISTORY_FIELDS if k.startswith("_stage_")),
                    "rear_barrier": tuple(k for k in HISTORY_FIELDS if k.startswith("_rear_barrier_"))}
    for state in states:
        seed, state_id = int(state["seed"]), str(state["state_id"])
        if (seed, state_id) in seen: raise DiagnosticError("Duplicate shared-state memory record")
        seen.add((seed, state_id))
        history = state.get("history_change", {})
        changed = history.get("changed_fields")
        if not isinstance(changed, list) or len(set(changed)) != len(changed) or any(k not in HISTORY_FIELDS for k in changed):
            raise DiagnosticError("Memory diagnostics need the actual changed-field list, not the full intervention whitelist")
        if "changed" in history and bool(history["changed"]) != bool(changed):
            raise DiagnosticError("Memory changed flag disagrees with changed fields")
        if "selected_changed" not in state or "score_changed" not in state:
            raise DiagnosticError("Memory record lacks selection/score change flags")
        full, reset = state.get("full", {}), state.get("reinitialized", {})
        if full.get("candidates") is not None and reset.get("candidates") is not None:
            first, second = full["candidates"], reset["candidates"]
            if len(first) != len(second) or [_identity(row["tasks"]) for row in first] != [_identity(row["tasks"]) for row in second]:
                raise DiagnosticError("Memory intervention candidates are not matched in order")
            score_changed = any(abs(_finite(a["score"], "full memory score") - _finite(b["score"], "reinitialized memory score")) > 1e-10 for a, b in zip(first, second))
            if score_changed != bool(state["score_changed"]): raise DiagnosticError("Memory score-change flag disagrees with stored scores")
            if (int(full["selected_index"]) != int(reset["selected_index"])) != bool(state["selected_changed"]):
                raise DiagnosticError("Memory selection-change flag disagrees with stored choices")
        row = dict(seed=seed, state_id=state_id, history_changed=int(bool(changed)),
                   score_changed=int(bool(state["score_changed"])), selection_changed=int(bool(state["selected_changed"])),
                   changed_fields=changed)
        for field in HISTORY_FIELDS: row[field] = int(field in changed)
        for name, fields in field_groups.items(): row[name] = int(bool(set(fields) & set(changed)))
        groups[seed].append(row); details.append(row)
    seed_rows = []
    metrics = ("history_changed", "score_changed", "selection_changed", *field_groups, *HISTORY_FIELDS)
    for seed, rows in sorted(groups.items()):
        row = dict(seed=seed, states=len(rows))
        for metric in metrics:
            row[metric + "_count"] = sum(r[metric] for r in rows)
            row[metric + "_rate"] = row[metric + "_count"] / len(rows)
        seed_rows.append(row)
    summary = dict(training_seed_count=len(seed_rows), states=len(states))
    for metric in metrics:
        values = [row[metric + "_rate"] for row in seed_rows]
        summary[metric + "_count"] = sum(row[metric + "_count"] for row in seed_rows)
        summary[metric + "_seed_mean_rate"] = statistics.mean(values)
        summary[metric + "_seed_sd_rate"] = statistics.stdev(values) if len(values) > 1 else None
        summary[metric + "_pooled_rate"] = summary[metric + "_count"] / len(states)
    return dict(schema="sparse-shared-state-memory-diagnostics", summary=summary, seed_results=seed_rows,
        state_results=details, inferential_intervals=False,
        scope="sampled common decision states only; selection changes do not establish complete episode mediation",
        grouping="actual changed fields; joint history intervention does not identify causal effects of individual field groups")
