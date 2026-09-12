#!/usr/bin/env python3
"""Build a tiny, self-contained SQLite registry for artifact smoke tests.

The real registry (``example.db``) is ~875 MB, holds models trained on millions
of rows, and is deliberately not in version control. That makes it useless for
answering the question a smoke test actually asks: *does this binary work?*

This script runs the ordinary training pipeline over a few hundred synthetic
rows using the ``sample_enum_transformer`` column reference, producing a small
database containing one published model with a real transformation pipeline
(log on the numeric feature and target, one-hot on the categorical).

Usage:
    python scripts/make_fixture_db.py <output_dir>

Writes:
    <output_dir>/fixture.db          registry + artifacts
    <output_dir>/dataset/sample.csv  synthetic training data
    <output_dir>/.env                env file the server binary reads
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXPERIMENT_ID = "smoke_fixture"
INTENT = "smoke_fixture_intent"
CATEGORIES = ["A", "B", "C", "D"]
N_ROWS = 400


def write_csv(path: pathlib.Path, seed: int = 42) -> None:
    """Synthetic data with a genuine signal, so the model beats a constant."""
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id,categorical,numerical,target"]
    for i in range(1, N_ROWS + 1):
        cat = CATEGORIES[i % len(CATEGORIES)]
        numerical = max(1.0, rng.gauss(500, 120))
        # multiplicative relationship -> log-space linear, which is what the
        # pipeline's log transform is designed to exploit
        factor = {"A": 2.0, "B": 3.0, "C": 5.0, "D": 7.0}[cat]
        target = max(1.0, numerical * factor * rng.gauss(1.0, 0.05))
        lines.append(f"{i},{cat},{numerical:.4f},{target:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_config(db_path: pathlib.Path, dataset_dir: pathlib.Path) -> dict:
    return {
        "name": INTENT,
        "description": "Synthetic fixture used by scripts/smoke_test.sh",
        "repository": {
            "experiment_id": EXPERIMENT_ID,
            "data": {"type": "sqlite", "properties": {"name": str(db_path), "migrate": True}},
            "object": {"type": "sqlite", "properties": {"name": str(db_path), "migrate": True}},
        },
        "instructions": [
            {
                "type": "data_io",
                "properties": {
                    "path": str(dataset_dir),
                    "file": "sample",
                    "format": "csv",
                    "reference": "sample_enum_transformer",
                },
                "call": [{"type": "load", "n_rows": N_ROWS}],
            },
            {
                "type": "data_cleaner",
                "properties": {"type": "lazy_call", "reference": "sample_enum_transformer"},
                "call": [
                    {
                        "type": "filter_columns",
                        "columns": ["column_categorical", "column_numerical", "column_target"],
                    },
                    {"type": "drop_na"},
                ],
            },
            {
                "type": "data_transformer",
                "properties": {"type": "lazy_call", "reference": "sample_enum_transformer"},
                "call": [
                    {
                        "type": "log_transformation",
                        "condition": "replace",
                        "columns": ["column_numerical", "column_target"],
                    },
                    {
                        "type": "one_hot_encoding",
                        "condition": "append_and_remove",
                        "columns": ["column_categorical"],
                    },
                ],
            },
            {
                "type": "model_trainer",
                "properties": {
                    "objective": "best_model",
                    "random_state": 42,
                    "fold": 5,
                    "parameter_grid": {"type": "exhaustive"},
                    "primary_metric": "mae",
                    "metrics": ["mae"],
                },
                "call": [{"model_type": "linear_regression", "hyperparameters": {}}],
            },
        ],
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: make_fixture_db.py <output_dir>", file=sys.stderr)
        return 2

    out_dir = pathlib.Path(argv[1]).resolve()
    dataset_dir = out_dir / "dataset"
    db_path = out_dir / "fixture.db"

    for stale in (db_path, out_dir / "dataset" / "sample.csv"):
        if stale.exists():
            stale.unlink()

    write_csv(dataset_dir / "sample.csv")

    # Imported after sys.path is fixed up, and after the CSV exists.
    from train.scenario_manager import InstructionFactory, ScenarioManager

    config = build_config(db_path, dataset_dir)
    instruction = InstructionFactory.parse_instruction(config)
    ScenarioManager(instruction).construct().execute()

    if not db_path.exists():
        print("ERROR: pipeline finished but no database was produced", file=sys.stderr)
        return 1

    # The server binary reads its configuration from .env in the working dir.
    (out_dir / ".env").write_text(
        "\n".join(
            [
                f"EXPERIMENT_ID={EXPERIMENT_ID}",
                "COLUMN_REFERENCE=sample_enum_transformer",
                f"REPOSITORY_DATA_PATH={db_path}",
                f"REPOSITORY_OBJECT_PATH={db_path}",
                "REPOSITORY_DATA=sqlite",
                "REPOSITORY_OBJECT=sqlite",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps({"db": str(db_path), "experiment_id": EXPERIMENT_ID, "intent": INTENT}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
