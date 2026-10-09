#!/usr/bin/env python3
"""Verify historical exported scalars, independently of models or simulator code.

Input JSON maps semantic input names to {path, sha256}. Paths are relative to
the input JSON, or absolute. All hashes are checked before arithmetic. No
manuscript transcription, synthetic episodes, model, or old runner is imported.

This authenticates packaged file identity and recalculates archived results;
it does not reconstruct missing receipt bodies or replay scientific episodes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import statistics

SEEDS = tuple(range(201, 209))
# Student-t quantiles for the declared eight-seed protocols (df=7).
# The family-size 3 and 5 values are Bonferroni simultaneous 95% intervals.
# Independently checked against scipy.stats.t.ppf; SciPy is not a runtime need.
T_CRITICAL_DF7 = {1: 2.3646242515927844, 3: 3.1275522742463706,
                  5: 3.4994832973504924}
SCORE_RULES = {
    "Full COGP": "learned_first_q",
    "Reward-only": "learned_first_scaled_reward",
    "Tail-only": "learned_first_discounted_tail_value",
}
REFERENCES = ("HCCG", "COGP_HCCG")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite archived scalar")
    return result


def truth(value):
    require(str(value).lower() in {"true", "false", "1", "0"},
            "Invalid boolean archived value")
    return str(value).lower() in {"true", "1"}


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        result = list(csv.DictReader(handle))
    require(result, "Empty required CSV: " + str(path))
    require(not any(truth(r.get("test_only", "false")) or truth(r.get("engineering_test", "false"))
                    or not truth(r.get("scientific_policy", "true")) for r in result),
            "Engineering/test-only records cannot authenticate scientific results")
    return result


def close(actual, expected, message, tolerance=1e-10):
    require(abs(actual - number(expected)) <= tolerance, message)


def paired_interval(values, family_size=1):
    require(len(values) == 8, "Declared inference needs all eight training seeds")
    require(family_size in T_CRITICAL_DF7, "Unrecognized declared comparison family")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(8)
    critical = T_CRITICAL_DF7[family_size]
    return {"seed_differences_pp": values, "mean_difference_pp": mean,
            "sample_sd_pp": sd, "standard_error_pp": se, "df": 7,
            "family_size": family_size, "t_critical": critical,
            "ci95_pp": [mean - critical * se, mean + critical * se],
            "positive_seeds": sum(value > 0 for value in values),
            "tied_seeds": sum(value == 0 for value in values)}


def memory_result(paths):
    episodes = rows(paths["memory_episodes"])
    detail = rows(paths["memory_detail"])
    study = json.loads(paths["memory_study"].read_text())
    worlds = study["worlds"]["test"]
    expected_cases = {world["case_id"] for world in worlds}
    require(len(expected_cases) == len(worlds) == 48, "Memory panel must have 48 exact worlds")
    require(tuple(study["protocol"]["seeds"]) == SEEDS, "Memory seed protocol differs")
    require(study["protocol"]["arms"] == ["full", "reinitialized"], "Memory arms differ")
    key = lambda row: (int(row["seed"]), row["arm"], row["case_id"])
    base = {key(row): row for row in episodes}
    richer = {key(row): row for row in detail}
    expected = {(seed, arm, case) for seed in SEEDS
                for arm in ("full", "reinitialized") for case in expected_cases}
    require(len(base) == len(episodes) == 768 and set(base) == expected,
            "Incomplete or duplicate memory episode records")
    require(len(richer) == len(detail) == 768 and set(richer) == expected,
            "Incomplete or duplicate detailed memory episode records")
    for identity, row in base.items():
        require(all(row[name] == richer[identity][name] for name in row),
                "Detailed export differs from episode export: " + str(identity))
        require(truth(richer[identity]["protected_flags_verified"]), "Recorded clone protection failed")
        require(len(richer[identity]["receipt_sha256"]) == 64, "Missing original receipt digest")
        require(row["success"] in {"0", "1"}, "Invalid binary success")
    for case in expected_cases:
        require(len({row["initial_hash"] for row in episodes if row["case_id"] == case}) == 1,
                "A fixed world changes its initial digest across methods or training seeds")
    per_seed = []
    contingency = Counter()
    for seed in SEEDS:
        counts = {arm: sum(int(base[seed, arm, case]["success"]) for case in expected_cases)
                  for arm in ("full", "reinitialized")}
        for case in expected_cases:
            full = base[seed, "full", case]
            reset = base[seed, "reinitialized", case]
            require(full["initial_hash"] == reset["initial_hash"], "Unmatched paired initial worlds")
            contingency[int(full["success"]), int(reset["success"])] += 1
        per_seed.append({"seed": seed, "episodes_per_arm": 48,
                         "full_successes": counts["full"],
                         "reinitialized_successes": counts["reinitialized"],
                         "full_percent": 100 * counts["full"] / 48,
                         "reinitialized_percent": 100 * counts["reinitialized"] / 48,
                         "difference_pp": 100 * (counts["full"] - counts["reinitialized"]) / 48})
    result = {"worlds": 48, "episodes_per_arm": 384, "training_seeds": list(SEEDS),
              "per_seed": per_seed,
              "paired_contrast": paired_interval([r["difference_pp"] for r in per_seed]),
              "contingency": {"both_success": contingency[1, 1], "both_failure": contingency[0, 0],
                              "full_only_success": contingency[1, 0], "reinitialized_only_success": contingency[0, 1]},
              "scope": "One prespecified seed-paired contrast on 48 fixed new worlds; exported scalars, not trajectory replay"}
    for arm in ("full", "reinitialized"):
        result[arm] = {"successes": sum(r[arm + "_successes"] for r in per_seed),
                       "mean_percent": statistics.mean(r[arm + "_percent"] for r in per_seed),
                       "sample_sd_percent": statistics.stdev(r[arm + "_percent"] for r in per_seed)}
    if "memory_common" in paths:
        common = rows(paths["memory_common"])
        common_key = lambda row: (int(row["seed"]), row["case_id"], int(row["decision_index"]))
        require(len(common) == 768 and len({common_key(row) for row in common}) == 768,
                "Common memory-state panel incomplete or duplicated")
        require({common_key(row) for row in common} ==
                {(seed, case, decision) for seed in SEEDS for case in expected_cases for decision in (0, 2)},
                "Common states differ from declared early-boundary sample")
        changed_scores = sum(any(abs(number(delta)) > 1e-9 for delta in
                                json.loads(row["q_delta_reinitialized_minus_full"])) for row in common)
        result["common_state_diagnostic"] = {
            "states": len(common), "memory_changed": sum(truth(r["memory_changed"]) for r in common),
            "scores_changed": changed_scores,
            "choice_changed": sum(truth(r["choice_disagreement"]) for r in common),
            "scope": "Shared states sampled only at decisions 0 and 2; does not fully explain closed-loop success differences"}
    return result


def score_results(paths):
    candidates = rows(paths["candidates"])
    require(len(candidates) == 726, "Expected complete 726-candidate export")
    grouped = defaultdict(list)
    for row in candidates:
        grouped[int(row["seed"]), row["case_id"], int(row["decision_index"])].append(row)
    require(len(grouped) == 192 and sorted({key[0] for key in grouped}) == [201, 208],
            "Expected 192 common states from the two archived seeds")
    per_state = []
    for (seed, case, decision), pool in sorted(grouped.items()):
        require(decision in (0, 2), "Unexpected diagnostic decision boundary")
        require(len({r["state_digest"] for r in pool}) == 1, "Mixed physical states in candidate group")
        require([int(r["candidate_index"]) for r in pool] == list(range(len(pool))),
                "Candidate order is not the original contiguous order")
        require(1 <= len(pool) <= 4, "Invalid effective candidate count")
        scores = {rule: [number(r[field]) for r in pool] for rule, field in SCORE_RULES.items()}
        for index in range(len(pool)):
            close(scores["Full COGP"][index], scores["Reward-only"][index] + scores["Tail-only"][index],
                  "Recorded full score is not reward plus terminal term", 1e-8)
        for reference in REFERENCES:
            returns = [number(r[reference + "_return_value"]) for r in pool]
            best = max(returns)
            winners = {i for i, value in enumerate(returns)
                       if math.isclose(value, best, rel_tol=1e-7, abs_tol=1e-9)}
            for rule, values in scores.items():
                chosen = max(range(len(pool)), key=lambda i: values[i])
                per_state.append({"reference_policy": reference, "rule": rule, "seed": seed,
                                  "case_id": case, "decision_index": decision, "candidate_count": len(pool),
                                  "selected_index": chosen, "regret": max(0.0, best - returns[chosen]),
                                  "best_agreement": int(chosen in winners),
                                  "selected_reference_return": returns[chosen]})
    per_seed = []
    for reference in REFERENCES:
        for rule in SCORE_RULES:
            for seed in (201, 208):
                sample = [r for r in per_state if (r["reference_policy"], r["rule"], r["seed"]) == (reference, rule, seed)]
                require(len(sample) == 96, "Each diagnostic seed must contribute 96 equal-weight states")
                per_seed.append({"reference_policy": reference, "rule": rule, "seed": seed,
                                 "states": len(sample), "regret": statistics.mean(r["regret"] for r in sample),
                                 "best_agreement": statistics.mean(r["best_agreement"] for r in sample),
                                 "selected_reference_return": statistics.mean(r["selected_reference_return"] for r in sample)})
    if "score_expected_seed" in paths:
        recorded = {(r["reference_policy"], r["rule"], int(r["seed"])): r
                    for r in rows(paths["score_expected_seed"])}
        require(len(recorded) == 12, "Incomplete archived component seed metrics")
        for row in per_seed:
            old = recorded[row["reference_policy"], row["rule"], row["seed"]]
            for field in ("regret", "best_agreement", "selected_reference_return"):
                close(row[field], old[field], "Component metric differs from server export")
    summary = []
    for reference in REFERENCES:
        for rule in SCORE_RULES:
            sample = [r for r in per_seed if r["reference_policy"] == reference and r["rule"] == rule]
            summary.append({"reference_policy": reference, "rule": rule,
                            "regret_mean": statistics.mean(r["regret"] for r in sample),
                            "best_agreement_mean": statistics.mean(r["best_agreement"] for r in sample),
                            "selected_reference_return_mean": statistics.mean(r["selected_reference_return"] for r in sample)})
    return {"candidate_records": 726, "states": 192, "seeds": [201, 208],
            "per_seed": per_seed, "summary": summary,
            "tie_break": "First exact maximum in original candidate order",
            "reference_best_set": "math.isclose(rel_tol=1e-7,abs_tol=1e-9)",
            "weighting": "Equal states within seed, then equal seed means",
            "native_identity_available": False,
            "scope": "Post-hoc descriptive candidate re-ranking only; reference continuation is not globally optimal; no new rollouts or closed-loop ablation"}


def detailed_history_result(paths):
    """Verify full exported decision arrays and the sparse matched pool records."""
    detail = rows(paths["memory_detail"])
    decisions = rows(paths["memory_decisions"])
    episodes = {(int(r["seed"]), r["arm"], r["case_id"]): r for r in detail}
    grouped = defaultdict(list)
    for row in decisions:
        key = (int(row["seed"]), row["arm"], row["case_id"])
        require(key in episodes, "Decision belongs to an unknown episode")
        require(row["receipt_sha256"] == episodes[key]["receipt_sha256"], "Decision receipt identity differs")
        pool = [number(x) for x in json.loads(row["candidate_q_in_original_order"])]
        require(len(pool) == int(row["actual_k"]) and len(pool) > 0, "Incomplete ordered decision scores")
        selected = int(row["selected_index"])
        require(selected == max(range(len(pool)), key=pool.__getitem__), "Recorded decision is not first score maximum")
        close(pool[selected], row["selected_q"], "Selected decision score differs")
        require(truth(row["score_components_present_all_candidates"]), "Decision lacks declared score components")
        require(truth(row["protected_flags_verified"]), "Recorded decision protection failed")
        grouped[key].append(int(row["decision_index"]))
    require(set(grouped) == set(episodes), "Missing full-trajectory decision export")
    for key, indices in grouped.items():
        require(sorted(indices) == list(range(int(episodes[key]["decision_steps"]))), "Missing or duplicate trajectory decision")
    common = [json.loads(line) for line in paths["memory_common_full"].read_text().splitlines() if line.strip()]
    declared = rows(paths["memory_common"])
    state_key = lambda r: (int(r["seed"]), r["case_id"], int(r["decision_index"]))
    lookup = {state_key(r): r for r in declared}
    require(len(common) == len(lookup) == 768 and {state_key(r) for r in common} == set(lookup), "Incomplete matched-state JSONL")
    changed_scores = changed_choices = changed_history = candidate_pairs = 0
    for state in common:
        baseline = lookup[state_key(state)]
        require(truth(state["protected_flags_verified"]), "Recorded shared-state protection failed")
        full, reset = state["full_candidates"], state["reinitialized_candidates"]
        require(len(full) == len(reset) == len(state["candidate_order"]) == int(state["candidate_count"]), "Matched candidate count differs")
        for pool in (full, reset):
            require([r["task_keys"] for r in pool] == state["candidate_order"], "Matched candidate order differs")
            for index, row in enumerate(pool):
                require(int(row["index"]) == index, "Candidate order index differs")
                close(.10 * number(row["reward"]), row["scaled_reward"], "Reward scale differs")
                discount = number(row["discount_gamma"]) ** int(row["predicted_high_steps"])
                close(discount, row["discount"], "Duration discount differs")
                tail = 0. if truth(row["terminal"]) else discount * number(row["tail_value"])
                close(tail, row["discounted_tail_value"], "Terminal score differs")
                close(row["policy_prior_coefficient"], 0., "Unexpected prior in history panel")
                close(number(row["scaled_reward"]) + tail, row["q"], "Candidate total score differs", tolerance=1e-8)
            selected_name = "full_selected_index" if pool is full else "reinitialized_selected_index"
            require(int(state[selected_name]) == max(range(len(pool)), key=lambda i: number(pool[i]["q"])), "Shared-state choice differs from first maximum")
        score_changed = any(abs(number(a["q"]) - number(b["q"])) > 1e-9 for a, b in zip(full, reset))
        choice_changed = int(state["full_selected_index"]) != int(state["reinitialized_selected_index"])
        memory_changed = bool(state["reinitialized_changed_fields"])
        require(score_changed == truth(state["score_changed"]), "Shared-state score change flag differs")
        require(choice_changed == truth(state["choice_disagreement"]) == truth(baseline["choice_disagreement"]), "Shared-state choice flag differs")
        require(memory_changed == truth(state["memory_changed"]) == truth(baseline["memory_changed"]), "Shared-state history flag differs")
        require(state["no_reference_return_in_this_experiment"] is True, "Memory and continuation diagnostic scopes must remain distinct")
        changed_scores += score_changed; changed_choices += choice_changed; changed_history += memory_changed
        candidate_pairs += len(full)
    return {"episode_records": len(episodes), "decision_records": len(decisions), "shared_states": len(common),
            "matched_candidate_pairs": candidate_pairs, "history_changed": changed_history,
            "score_changed": changed_scores, "choice_changed": changed_choices,
            "checks": ["receipt identity", "complete decision order", "first-maximum choices", "shared candidate order", "reward/discount/terminal decomposition"],
            "scope": "Consistency of exported records and recorded protection flags; no simulator snapshot or historical trajectory replay"}


def control_results(paths):
    seeds = rows(paths["controls_seeds"])
    cells = rows(paths["controls_cells"])
    published = {(row["phase"], row["arm"]): row for row in rows(paths["controls_summary"])}
    study = json.loads(paths["controls_study"].read_text())
    by_arm = defaultdict(dict)
    for row in seeds:
        identity = (row["phase"], row["arm"])
        seed = int(row["seed"])
        require(seed not in by_arm[identity], "Duplicate aggregate seed row")
        observed = int(row["observed_episodes"])
        success = int(row["observed_successes"])
        require(observed == int(row["expected_episodes"]) and 0 <= success <= observed,
                "Incomplete aggregate seed panel")
        by_arm[identity][seed] = 100 * success / observed
        close(by_arm[identity][seed], row["observed_success_percent"], "Seed percentage differs from integer count")
    require(set(by_arm) == set(published), "Seed aggregate arms differ from archived complete summary")
    summary = []
    for identity, sample in sorted(by_arm.items()):
        arm_seeds = [row for row in seeds if (row["phase"], row["arm"]) == identity]
        arm_cells = [row for row in cells if (row["phase"], row["arm"]) == identity]
        require(all(int(r["expected_episodes"]) == int(r["observed_episodes"]) for r in arm_cells),
                "Incomplete aggregate condition panel")
        total = sum(int(r["observed_episodes"]) for r in arm_seeds)
        success = sum(int(r["observed_successes"]) for r in arm_seeds)
        require(total == sum(int(r["observed_episodes"]) for r in arm_cells) and
                success == sum(int(r["successes"]) for r in arm_cells), "Seed/cell integer totals disagree")
        row = {"phase": identity[0], "arm": identity[1], "episodes": total,
               "successes": success, "training_seeds": len(sample),
               "mean_percent": statistics.mean(sample.values()),
               "sample_seed_sd_pp": statistics.stdev(sample.values()) if len(sample) > 1 else None}
        old = published[identity]
        require(total == int(old["observed_episodes"]) and success == int(old["observed_successes"]),
                "Published aggregate count differs from source counts")
        close(row["mean_percent"], old["mean_observed_seed_success_percent"], "Aggregate mean differs")
        if row["sample_seed_sd_pp"] is not None:
            close(row["sample_seed_sd_pp"], old["observed_seed_sd_pp"], "Aggregate SD differs")
        summary.append(row)
    declared = []
    expected_contrasts = {r["contrast"]: r for r in rows(paths["controls_contrasts"])}
    for phase, field, family in (("core", "core_contrasts", "core5"),
                                  ("actor_eval", "actor_contrasts", "actor3")):
        definitions = study["protocol"][field]
        require(len(definitions) == (5 if phase == "core" else 3), "Declared comparison family differs")
        for name, coefficients in definitions:
            require(all(set(by_arm[phase, arm]) == set(SEEDS) for arm in coefficients),
                    "All contrast arms need all eight matched seeds")
            values = [sum(coefficient * by_arm[phase, arm][seed]
                          for arm, coefficient in coefficients.items()) for seed in SEEDS]
            result = dict(contrast=name, family=family, coefficients=coefficients,
                          **paired_interval(values, len(definitions)))
            old = expected_contrasts[name]
            close(result["mean_difference_pp"], old["difference_pp"], "Contrast mean differs")
            close(result["sample_sd_pp"], old["seed_difference_sd_pp"], "Contrast SD differs")
            close(result["ci95_pp"][0], old["lower_pp"], "Contrast interval lower bound differs")
            close(result["ci95_pp"][1], old["upper_pp"], "Contrast interval upper bound differs")
            declared.append(result)
    stress = defaultdict(lambda: [0, 0])
    for row in cells:
        if row["phase"] == "stress":
            entry = stress[row["arm"], row["physics_multiplier"]]
            entry[0] += int(row["successes"])
            entry[1] += int(row["observed_episodes"])
    return {"seed_rows": len(seeds), "cell_rows": len(cells), "arm_summary": summary,
            "prespecified_contrasts": declared,
            "stress_by_multiplier": [{"arm": arm, "multiplier": multiplier,
                                      "successes": count[0], "episodes": count[1],
                                      "success_percent": 100 * count[0] / count[1],
                                      "inference": "descriptive; no prespecified pressure-test family"}
                                     for (arm, multiplier), count in sorted(stress.items())],
            "scope": "Integer aggregate arithmetic and eight-seed contrast recomputation; original core/actor/stress episode receipt bodies are not included"}


def timing_result(paths):
    latency = [r for r in rows(paths["controls_latency"]) if r["run_scope"] == "exclusive_timing"]
    require(len(latency) == 144, "Exclusive latency panel must have 3 methods x 48 worlds")
    by_arm = defaultdict(list)
    for row in latency:
        require(row["phase"] == "timing" and int(row["training_seeds"]) == 1,
                "Timing records have unexpected scope")
        by_arm[row["arm"]].append(row)
    require(len(by_arm) == 3, "Expected all three planned generators in timing panel")
    result = []
    for arm, sample in sorted(by_arm.items()):
        require(len(sample) == 48, "Timing method must contain 48 full-episode condition summaries")
        require(len({tuple(r[field] for field in
                           ("N", "M", "kind", "workload", "physics_multiplier")) for r in sample}) == 48,
                "Duplicate exclusive-timing condition summary")
        decisions = sum(int(r["decisions"]) for r in sample)
        result.append({"arm": arm, "episodes": sum(int(r["episodes"]) for r in sample),
                       "decisions": decisions,
                       "decision_weighted_mean_seconds": sum(int(r["decisions"]) * number(r["decision_weighted_mean_seconds"]) for r in sample) / decisions,
                       "maximum_seconds": max(number(r["maximum_seconds"]) for r in sample)})
    return {"summary": result, "total_decisions": sum(row["decisions"] for row in result),
            "scope": "One training seed; 48 worlds per method, full-trajectory selection summaries; descriptive only",
            "limitation": "Per-cell latency quantiles cannot reconstruct pooled N,M quantiles; no fixed wall-clock budget comparison"}


def verify_inputs(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    paths, verified = {}, []
    for name, entry in manifest.items():
        path = Path(entry["path"])
        if not path.is_absolute(): path = manifest_path.parent / path
        require(path.is_file(), "Missing packaged historical input: " + name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == entry["sha256"], "Historical input digest differs: " + name)
        paths[name] = path
        verified.append({"input": name, "sha256": digest, "bytes": path.stat().st_size})
    return paths, verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parents[1]/"assets"/"historical_inputs.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"verification"/"historical_results.json")
    args = parser.parse_args()
    paths, verified = verify_inputs(args.inputs)
    result = {"status": "PASS", "verified_inputs": verified,
              "execution_memory": memory_result(paths), "score_components": score_results(paths),
              "controls": control_results(paths), "exclusive_timing": timing_result(paths),
              "detailed_execution_history": detailed_history_result(paths),
              "new_training": False, "new_simulator_rollouts": False,
              "method": "Independent offline arithmetic from actual historical returned data, without old experiment imports",
              "boundary": "File hashes and numerical reconciliation do not replace scientific trajectory replay with original model bytes"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "memory": result["execution_memory"]["paired_contrast"],
                      "score_components": result["score_components"]["summary"],
                      "exclusive_timing": result["exclusive_timing"]}, indent=2))


if __name__ == "__main__":
    main()
