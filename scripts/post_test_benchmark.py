#!/usr/bin/env python3
"""Re-measure a stored run's post_test error at a controlled sample size.

`post_test` reads its sample size from the step's `properties`, but every config
in `train_config/` writes the count into the step's `call` as `n_samples`. The
loader therefore falls back to its default of 1000 rows, while the run records
`size.post_test.row` from the call -- so the registry claims 100,000 and the
metric came from ~700.

That makes recorded `validation.post_test.mae` values both noisy and
incomparable. This harness rebuilds the inference machine from the stored
artifacts, samples a controlled number of rows, applies the same cleaner the
run trained with, and reports the error.

Usage:
    python scripts/post_test_benchmark.py <run_id> <train_config.json> [--n-rows N] [--seed S]

Example:
    python scripts/post_test_benchmark.py 73 train_config/beat_benchmark_1m.json --n-rows 100000
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import runtime_config  # noqa: E402
from repositories.repo import Facade  # noqa: E402
from train.data_cleaner import Cleaner  # noqa: E402
from train.post_test import PostTest  # noqa: E402


def load_config(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def build(run_id: int, config_path: str, n_rows: int, seed: int):
    cfg = load_config(config_path)
    steps = {s["type"]: s for s in cfg["instructions"]}
    if "post_test" not in steps:
        raise SystemExit(f"{config_path} has no post_test step")

    facade = Facade.parse_instruction(runtime_config.load())
    cleaner = Cleaner.parse_instruction(
        steps["data_cleaner"]["properties"], steps["data_cleaner"]["call"], facade
    )
    # Harness run: never let a measurement write into the registry.
    cleaner.facade = None

    props = dict(steps["post_test"]["properties"])
    props["n_rows"] = n_rows
    props["random_state"] = seed
    post_test = PostTest.parse_instruction(props, steps["post_test"]["call"], cleaner, facade)
    return post_test


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id", type=int, help="parent run id holding the model")
    ap.add_argument("config", help="the train_config the run was produced from")
    ap.add_argument("--n-rows", type=int, default=100_000, help="rows to sample (default 100000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    post_test = build(args.run_id, args.config, args.n_rows, args.seed)

    started = time.time()
    samples = post_test.pick_random_samples(args.n_rows, args.seed)
    inference = post_test.reconstruct_inference(args.run_id)
    scores = post_test.check(inference, samples, ["mae"])
    elapsed = time.time() - started

    print(f"run {args.run_id}  ({args.config})")
    print(f"  sampled rows            {args.n_rows:>9,}")
    print(f"  after the run's cleaner {len(samples):>9,}")
    print(f"  post_test MAE           ${list(scores.values())[0]:>13,.2f}")
    print(f"  elapsed                 {elapsed:>9.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
