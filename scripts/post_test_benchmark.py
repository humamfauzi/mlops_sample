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
    ap.add_argument("--seed", type=int, default=42, help="single draw; ignored with --seeds")
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        metavar="S",
        help="draw several samples and report the spread (recommended; see below)",
    )
    args = ap.parse_args(argv)

    seeds = args.seeds if args.seeds else [args.seed]
    post_test = build(args.run_id, args.config, args.n_rows, seeds[0])
    inference = post_test.reconstruct_inference(args.run_id)

    started = time.time()
    scores = []
    rows = []
    for seed in seeds:
        samples = post_test.pick_random_samples(args.n_rows, seed)
        result = post_test.check(inference, samples, ["mae"])
        scores.append(list(result.values())[0])
        rows.append(len(samples))
    elapsed = time.time() - started

    print(f"run {args.run_id}  ({args.config})")
    print(f"  rows drawn per sample   {args.n_rows:>9,}")
    print(f"  rows after the cleaner  {min(rows):>9,}", end="")
    print("" if min(rows) == max(rows) else f" .. {max(rows):,}")
    print()

    if len(scores) == 1:
        print(f"  post_test MAE           ${scores[0]:>13,.2f}")
        print()
        print("  A single draw is not a measurement. The same model on different")
        print("  100k samples of CFS 2017 varies by roughly 40%, because ten")
        print("  shipments out of 100,000 can decide a quarter of this metric.")
        print("  Re-run with --seeds 42 7 101 2024 31337 for a usable range.")
    else:
        ordered = sorted(scores)
        median = ordered[len(ordered) // 2]
        print(f"  post_test MAE           ${median:>13,.2f}   (median of {len(scores)})")
        print(f"    min ${min(scores):>13,.2f}")
        print(f"    max ${max(scores):>13,.2f}")
        print(f"    spread            ${max(scores) - min(scores):>13,.2f}"
              f"   ({100 * (max(scores) - min(scores)) / median:.0f}% of the median)")
        if max(scores) - min(scores) > 0.2 * median:
            print()
            print("  This spread is wider than any improvement worth chasing.")
            print("  Compare candidates on the median of several draws, or on")
            print("  log-space test MAE, which is stable to about 0.005.")

    print(f"\n  elapsed                 {elapsed:>9.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
