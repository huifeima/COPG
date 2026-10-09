"""Fit one shared-interface dynamics model from an acceleration archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fit import fit
from .models import MODEL_KINDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=MODEL_KINDS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    arguments = parser.parse_args()
    geometry = json.loads(arguments.geometry.read_text()) if arguments.geometry else {}
    report = fit(arguments.data, arguments.output, arguments.kind, geometry,
                 arguments.seed, updates=arguments.updates, batch_size=arguments.batch_size,
                 device=arguments.device, split_manifest=arguments.split_manifest)
    print(json.dumps({key: report[key] for key in
                      ("status", "parameters", "best_step", "best_validation_force_rmse",
                       "checkpoint_sha256", "training_samples", "validation_samples")}, indent=2))


if __name__ == "__main__":
    main()
