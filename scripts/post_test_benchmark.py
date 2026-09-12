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

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import runtime_config  # noqa: E402
from repositories.repo import Facade  # noqa: E402
from train.data_cleaner import Cleaner  # noqa: E402
from train.post_test import (  # noqa: E402
    PostTest,
    calibration_by_decile,
    value_weighted_log_mae,
)


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
    ap.add_argument("--seed", type=int, default=42, help="single draw; ignored with --seeds")
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        metavar="S",
        help="draw several samples and report the spread (recommended; see below)",
    )
    ap.add_argument(
        "--calibration",
        action="store_true",
        help="also print mean actual vs mean predicted per value decile",
    )
    ap.add_argument(
        "--cap",
        type=float,
        default=1_000_000.0,
        help="weight cap for the value-weighted log error (default 1e6)",
    )
    args = ap.parse_args(argv)

    seeds = args.seeds if args.seeds else [args.seed]
    post_test = build(args.run_id, args.config, args.n_rows, seeds[0])
    inference = post_test.reconstruct_inference(args.run_id)

    started = time.time()
    dollar, vwle, rows = [], [], []
    last = None
    for seed in seeds:
        samples = post_test.pick_random_samples(args.n_rows, seed)
        actual = samples[post_test.column.target()].to_numpy(dtype=float)
        predicted = post_test.predict(inference, samples)
        dollar.append(float(np.abs(predicted - actual).mean()))
        vwle.append(value_weighted_log_mae(actual, predicted, cap=args.cap))
        rows.append(len(samples))
        last = (actual, predicted)
    elapsed = time.time() - started

    print(f"run {args.run_id}  ({args.config})")
    print(f"  rows drawn per sample   {args.n_rows:>9,}")
    print(f"  rows after the cleaner  {min(rows):>9,}", end="")
    print("" if min(rows) == max(rows) else f" .. {max(rows):,}")
    print()

    def spread(values):
        return 100 * (max(values) - min(values)) / np.median(values)

    print(f"  {'metric':<26}{'median':>14}{'spread':>10}   note")
    print(f"  {'dollar MAE':<26}{'$' + format(np.median(dollar), ',.2f'):>14}"
          f"{spread(dollar):>9.0f}%   headline only")
    print(f"  {'value-weighted log MAE':<26}{np.median(vwle):>14,.4f}"
          f"{spread(vwle):>9.0f}%   dollar-aligned, weight capped at ${args.cap:,.0f}")

    if len(dollar) == 1:
        print()
        print("  A single draw is not a measurement. Dollar MAE varies by roughly")
        print("  40% across 100k samples of CFS 2017, because ten shipments out of")
        print("  100,000 can decide a quarter of it. Re-run with")
        print("  --seeds 42 7 101 2024 31337 for the range, and prefer the")
        print("  value-weighted log MAE, which is roughly five times tighter.")

    if args.calibration and last is not None:
        actual, predicted = last
        print(f"\n  calibration by value decile (seed {seeds[-1]}):")
        print(f"    {'decile':<8}{'value range':>30}{'rows':>8}{'log MAE':>9}"
              f"{'mean actual':>15}{'mean pred':>15}{'ratio':>8}")
        for r in calibration_by_decile(actual, predicted, bins=10):
            rng = f"${r['value_min']:,.0f}-${r['value_max']:,.0f}"
            print(f"    {r['decile']:<8}{rng:>30}{r['rows']:>8,}{r['log_mae']:>9.3f}"
                  f"{r['mean_actual']:>15,.0f}{r['mean_predicted']:>15,.0f}{r['ratio']:>8.2f}")
        print("    ratio < 1 means the model under-predicts that decile")

    print(f"\n  elapsed                 {elapsed:>9.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
