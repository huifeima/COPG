"""Paper experiment definitions and explicit world manifests.

This module uses only the Python standard library. It never locates an older
project, imports a server runner, generates missing historical worlds silently,
or substitutes model weights. New worlds require an explicit generation call.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_REQUIRED_WORLD_FIELDS = ("case_id", "seed", "kind", "scenario", "dogs", "sheep", "family")

class ProtocolError(ValueError):
    """An experiment prerequisite or recorded protocol is invalid."""

def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProtocolError(f"Cannot read configuration {path}: {error}") from error

def list_panels(config_dir: str | Path | None = None) -> list[str]:
    root = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    return sorted(path.stem for path in (root / "panels").glob("*.json"))

def load_panel(name: str, config_dir: str | Path | None = None) -> dict[str, Any]:
    root = (Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR).resolve()
    if name not in list_panels(root):
        raise ProtocolError(f"Unknown panel {name!r}. Available: {', '.join(list_panels(root))}")
    panel = _read(root / "panels" / (name + ".json"))
    if panel.get("schema") != "experiment_panel" or panel.get("name") != name:
        raise ProtocolError(f"Panel schema/name mismatch: {name}")
    panel = copy.deepcopy(panel)
    panel["config_dir"] = str(root)
    seeds = panel.get("training_seeds")
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ProtocolError(f"Invalid training seeds for {name}")
    for arm_name, arm in panel.get("arms", {}).items():
        if not arm_name or any(token in arm_name for token in ("/", "\\", "..")):
            raise ProtocolError(f"Unsafe arm name: {arm_name}")
        if any(type(arm.get(field)) is not int or arm[field] <= 0 for field in ("K", "H", "E")):
            raise ProtocolError(f"Invalid planning budgets in {arm_name}")
        if not math.isfinite(float(arm.get("prior", 0.0))):
            raise ProtocolError(f"Nonfinite prior in {arm_name}")
        if arm.get("dynamics") == "true_diagnostic_only" and panel.get("kind") != "diagnostic":
            raise ProtocolError("True simulator dynamics are allowed only for offline diagnostic panels")
    return panel

def resolve_manifest(panel: dict[str, Any], world_manifest: str | Path | None = None) -> Path:
    if world_manifest is not None:
        return Path(world_manifest).expanduser().resolve()
    value = panel.get("world_manifest")
    if not value:
        raise ProtocolError(
            f"Panel {panel['name']} requires its world manifest, which is not supplied. "
            "Import the actual archived manifest for historical replay, or explicitly generate "
            "a new manifest and label the resulting run as a new experiment."
        )
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(panel.get("config_dir", DEFAULT_CONFIG_DIR)) / path).resolve()

def load_worlds(panel: dict[str, Any], world_manifest: str | Path | None = None) -> list[dict[str, Any]]:
    path = resolve_manifest(panel, world_manifest)
    worlds = _read(path)
    if not isinstance(worlds, list) or not worlds:
        raise ProtocolError(f"World manifest must be a nonempty list: {path}")
    expected = panel.get("worlds")
    if expected is not None and len(worlds) != expected:
        raise ProtocolError(f"{panel['name']} requires {expected} worlds/case variants; {path} contains {len(worlds)}")
    identities, physical_cases, condition_counts = set(), set(), {}
    condition_path = Path(panel.get("config_dir", DEFAULT_CONFIG_DIR)) / "conditions.json"
    allowed_conditions = None
    if condition_path.is_file():
        allowed_conditions = {(c["sheep"], c["dogs"], c["kind"], c["scenario"]) for c in _read(condition_path)}
    for world in worlds:
        missing = [field for field in _REQUIRED_WORLD_FIELDS if field not in world]
        if missing:
            raise ProtocolError(f"World is missing {missing}: {world.get('case_id')}")
        case_id = world["case_id"]
        if not isinstance(case_id, str) or not case_id or any(x in case_id for x in ("/", "\\", "..")) or case_id in identities:
            raise ProtocolError(f"Invalid or duplicate world identity: {case_id}")
        identities.add(case_id)
        for field in ("seed", "dogs", "sheep"):
            if type(world[field]) is not int or world[field] < (0 if field == "seed" else 1):
                raise ProtocolError(f"Invalid {field} in {case_id}")
        factor = float(world.get("physics_multiplier", 1.0))
        if not math.isfinite(factor) or factor <= 0.0:
            raise ProtocolError(f"Invalid physics multiplier in {case_id}")
        key = (world["seed"], world["sheep"], world["dogs"], world["kind"], world["scenario"], factor)
        if key in physical_cases:
            raise ProtocolError(f"Duplicate physical case: {case_id}")
        physical_cases.add(key)
        cell = (world["sheep"], world["dogs"], world["kind"], world["scenario"])
        if allowed_conditions is not None and cell not in allowed_conditions:
            raise ProtocolError(f"Unknown evaluation condition in {case_id}: {cell}")
        condition = (*cell, factor)
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
    declared_conditions = panel.get("conditions")
    if declared_conditions is not None and len(condition_counts) != declared_conditions:
        raise ProtocolError(f"{panel['name']} requires {declared_conditions} conditions; manifest has {len(condition_counts)}")
    repetitions = panel.get("worlds_per_condition")
    if repetitions is not None and any(count != repetitions for count in condition_counts.values()):
        raise ProtocolError(f"{panel['name']} requires {repetitions} worlds in every condition")
    return copy.deepcopy(worlds)

def make_jobs(panel: dict[str, Any] | str, output_root: str | Path, world_manifest: str | Path | None = None,
              config_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Create a deterministic finite job list without performing any experiment."""
    if isinstance(panel, str):
        panel = load_panel(panel, config_dir)
    root = Path(output_root).expanduser().resolve()
    name = panel["name"]
    if panel.get("kind") == "offline_analysis":
        return [dict(kind="offline_analysis", panel=name, path=str(root / name / "analysis.json"),
                     candidate_records=panel["candidate_records"], protocol=copy.deepcopy(panel))]
    worlds = load_worlds(panel, world_manifest)
    output = []
    for world in worlds:
        for seed in panel["training_seeds"]:
            if name == "cross_generator":
                arm_name = "cross_generator_cohort"
                output.append(dict(kind="diagnostic", panel=name, seed=seed, training_seed=seed,
                    arm=arm_name, arm_config=copy.deepcopy(panel["arms"]["hccg"]),
                    cohort_arms=copy.deepcopy(panel["arms"]), diagnostic_protocol={key:copy.deepcopy(panel[key])
                        for key in ("driver_states", "regimes", "matched_effective_candidates", "common_geometry_width", "reference_continuation") if key in panel},
                    case=copy.deepcopy(world), path=str(root / name / "receipts" / arm_name / f"seed_{seed}" / (world["case_id"] + ".json")),
                    manifest_override=world_manifest is not None, exclusive_worker=False))
                continue
            for arm_name, arm in panel["arms"].items():
                job_kind = panel.get("kind", "closed_loop")
                if job_kind == "closed_loop":
                    job_kind = "episode"
                output.append(dict(kind=job_kind, panel=name, seed=seed, training_seed=seed,
                    arm=arm_name, arm_config=copy.deepcopy(arm), case=copy.deepcopy(world),
                    path=str(root / name / "receipts" / arm_name / f"seed_{seed}" / (world["case_id"] + ".json")),
                    manifest_override=world_manifest is not None,
                    exclusive_worker=bool(panel.get("exclusive_worker", False))))
    return output

def known_world_seeds(config_dir: str | Path | None = None) -> set[int]:
    root = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    result = set()
    for path in (root / "manifests").rglob("*.json"):
        records = _read(path)
        if isinstance(records, list):
            result.update(row["seed"] for row in records if isinstance(row, dict) and type(row.get("seed")) is int)
    return result

def generate_worlds(name: str, base_seed: int, worlds_per_condition: int, output_manifest: str | Path,
                    config_dir: str | Path | None = None) -> dict[str, Any]:
    """Explicitly create NEW reset inputs, never recovered historical worlds.

    World addresses are ``base_seed + cell_index * 1000 + world_index``. The
    rule belongs to this new project, not the unavailable original training
    reset stream. Stress variants reuse each initial seed across three forces.
    """
    panel = load_panel(name, config_dir)
    if type(base_seed) is not int or base_seed < 0 or type(worlds_per_condition) is not int or not 1 <= worlds_per_condition < 1000:
        raise ProtocolError("base_seed must be nonnegative and worlds_per_condition between 1 and 999")
    if panel.get("kind") == "offline_analysis":
        raise ProtocolError("Offline score analysis does not create simulator worlds")
    conditions = _read(Path(panel["config_dir"]) / "conditions.json")
    stress = "physics_multipliers" in panel
    if stress:
        conditions = [c for c in conditions if c["sheep"] == 15 and c["kind"] == "scaling"]
    factors = panel.get("physics_multipliers", [1.0])
    worlds = []
    for i, cell in enumerate(conditions):
        for j in range(worlds_per_condition):
            for factor in factors:
                worlds.append(dict(cell, seed=base_seed+i*1000+j, case_id=f"{name}_new_c{i:02d}_w{j:03d}_f{factor:g}",
                                   condition=f"N{cell['sheep']}", physics_multiplier=float(factor)))
    overlap = {w["seed"] for w in worlds} & known_world_seeds(panel["config_dir"])
    if overlap:
        raise ProtocolError(f"New world addresses overlap supplied manifests: {sorted(overlap)[:5]}")
    target = Path(output_manifest).expanduser().resolve()
    if target.exists():
        raise ProtocolError(f"Refusing to overwrite a manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(worlds, indent=2) + "\n", encoding="utf-8")
    metadata = dict(schema="new_world_panel", panel=name, manifest=str(target), worlds=len(worlds),
                    base_seed=base_seed, worlds_per_condition=worlds_per_condition, exact_historical_replay=False,
                    addressing_rule="base_seed + cell_index * 1000 + world_index", source="explicit new-project generation",
                    independence_scope="no overlap with provided reset manifests; unavailable historical manifests not checked")
    target.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata

# Simple names for callers that use panel configuration terminology.
get_panel = load_panel
jobs = make_jobs
