"""Tests for post_test configuration and sampling.

Phase A of EXPERIMENT_PLAN.md. `PostTest` read its sample size from the step's
`properties`, but every config in `train_config/` writes the count into the
step's `call` as `n_samples`. Nothing set `properties.n_rows`, so the loader
always took its default of 1000 rows while the run recorded
`size.post_test.row = 100000` from the call.

The effect was not subtle: the published champion's error was recorded as
$7,667 on 686 rows, against $9,098 on the 69,944 rows the config asked for.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from train.post_test import (
    DEFAULT_N_SAMPLES,
    DEFAULT_SEED,
    Config,
    PostTest,
    calibration_by_decile,
    value_weighted_log_mae,
)


class RecordingLoader:
    """Captures how the sample was requested instead of reading a CSV."""

    def __init__(self):
        self.requests = []

    def load_random_rows_via_csv(self, column=None, n_rows=None, random_state=None,
                                 load_options=None):
        self.requests.append({"n_rows": n_rows, "random_state": random_state})
        return self

    def execute(self, _):
        return pd.DataFrame({"COLUMN_TARGET": [1.0, 2.0, 3.0]})


class PassthroughCleaner:
    """Stands in for Cleaner; records how it was asked to clean."""

    def __init__(self):
        self.calls = []

    def execute(self, frame):
        self.calls.append({"apply_population_filters": True, "record_metadata": True})
        return frame

    def clean_data(self, frame, apply_population_filters=True, record_metadata=True):
        self.calls.append({
            "apply_population_filters": apply_population_filters,
            "record_metadata": record_metadata,
        })
        return frame


def make_post_test(call, properties=None, loader=None):
    props = {"reference": "sample_enum_transformer", "path": "dataset", "file": "cfs_2017"}
    props.update(properties or {})
    return PostTest.parse_instruction(
        props, call, PassthroughCleaner(), None
    ) if loader is None else _with_loader(props, call, loader)


def _with_loader(props, call, loader):
    pt = PostTest.parse_instruction(props, call, PassthroughCleaner(), None)
    pt.loader = loader
    return pt


class TestSampleSizeResolution:
    def test_n_samples_comes_from_the_call(self):
        pt = make_post_test([{"n_samples": 100000, "metrics": ["mae"], "seed": 42}])

        assert pt.configs[0].n_samples == 100000

    def test_seed_comes_from_the_call(self):
        pt = make_post_test([{"n_samples": 10, "metrics": ["mae"], "seed": 7}])

        assert pt.configs[0].seed == 7

    def test_call_wins_over_properties(self):
        pt = make_post_test(
            [{"n_samples": 500, "metrics": ["mae"], "seed": 1}],
            properties={"n_rows": 99999, "random_state": 99},
        )

        assert pt.configs[0].n_samples == 500
        assert pt.configs[0].seed == 1

    def test_properties_are_a_fallback(self):
        # Configs written the other way round keep working.
        pt = make_post_test(
            [{"metrics": ["mae"]}],
            properties={"n_rows": 12345, "random_state": 3},
        )

        assert pt.configs[0].n_samples == 12345
        assert pt.configs[0].seed == 3

    def test_defaults_when_neither_is_given(self):
        pt = make_post_test([{"metrics": ["mae"]}])

        assert pt.configs[0].n_samples == DEFAULT_N_SAMPLES
        assert pt.configs[0].seed == DEFAULT_SEED

    def test_every_config_in_the_repository_resolves_its_n_samples(self):
        # The regression was invisible because the default looked plausible.
        # Assert the actual configs on disk carry the size they intend.
        root = pathlib.Path(__file__).resolve().parent.parent / "train_config"
        checked = 0
        for path in sorted(root.glob("*.json")):
            cfg = json.loads(path.read_text())
            for step in cfg["instructions"]:
                if step["type"] != "post_test":
                    continue
                pt = make_post_test(
                    step["call"], properties=step.get("properties", {})
                )
                intended = step["call"][0].get("n_samples", DEFAULT_N_SAMPLES)
                assert pt.configs[0].n_samples == intended, path.name
                checked += 1
        assert checked > 0, "no post_test steps found to check"


class TestSampling:
    def test_sampling_uses_the_configured_size_and_seed(self):
        loader = RecordingLoader()
        pt = make_post_test(
            [{"n_samples": 4321, "metrics": ["mae"], "seed": 11}], loader=loader
        )

        pt.pick_random_samples()

        assert loader.requests == [{"n_rows": 4321, "random_state": 11}]

    def test_explicit_arguments_override_the_config(self):
        loader = RecordingLoader()
        pt = make_post_test(
            [{"n_samples": 100, "metrics": ["mae"], "seed": 1}], loader=loader
        )

        pt.pick_random_samples(999, 55)

        assert loader.requests == [{"n_rows": 999, "random_state": 55}]

    def test_parse_instruction_does_not_sample(self):
        # Sampling at parse time meant the size had to be decided before the
        # per-config values were known.
        loader = RecordingLoader()
        make_post_test([{"n_samples": 10, "metrics": ["mae"]}], loader=loader)

        assert loader.requests == []


class TestCheckAgainst:
    def test_unknown_value_is_rejected(self):
        # It was parsed and never used; a typo passed silently.
        with pytest.raises(ValueError, match="check_against"):
            make_post_test([{"check_against": "yesterday", "metrics": ["mae"]}])

    def test_both_known_values_are_accepted(self):
        for value in ("actual_value", "random"):
            pt = make_post_test([{"check_against": value, "metrics": ["mae"]}])
            assert pt.configs[0].check_against == value

    def test_the_value_every_config_actually_uses_is_accepted(self):
        root = pathlib.Path(__file__).resolve().parent.parent / "train_config"
        for path in sorted(root.glob("*.json")):
            cfg = json.loads(path.read_text())
            for step in cfg["instructions"]:
                if step["type"] == "post_test":
                    for entry in step["call"]:
                        make_post_test([entry])


class TestCallValidation:
    def test_empty_call_is_rejected(self):
        with pytest.raises(ValueError, match="at least one entry"):
            make_post_test([])


class TestValueWeightedLogMae:
    """The dollar-aligned, stable alternative to raw dollar MAE.

    Dollar error is approximately value x relative error, so weighting the
    relative error by value tracks dollars -- but the raw weight is dominated by
    a handful of enormous shipments, which is why it is capped.
    """

    def test_separates_cases_that_log_mae_cannot(self):
        # Two shipments. Both models are 2x off on exactly one of them, so their
        # log MAE is identical -- but model B is off on the $100,000 one.
        y = np.array([10.0, 100_000.0])
        a = np.array([20.0, 100_000.0])
        b = np.array([10.0, 200_000.0])

        assert np.isclose(np.abs(np.log(a / y)).mean(), np.abs(np.log(b / y)).mean())
        assert np.abs(a - y).mean() < np.abs(b - y).mean()
        assert value_weighted_log_mae(y, a) < value_weighted_log_mae(y, b)

    def test_is_zero_for_a_perfect_prediction(self):
        y = np.array([1.0, 100.0, 10_000.0])

        assert value_weighted_log_mae(y, y) == pytest.approx(0.0)

    def test_weighting_favours_accuracy_on_large_values(self):
        y = np.array([10.0, 100_000.0])
        # same log error magnitude, applied to the small vs the large shipment
        off_small = np.array([20.0, 100_000.0])
        off_large = np.array([10.0, 200_000.0])

        assert value_weighted_log_mae(y, off_small) < value_weighted_log_mae(y, off_large)

    def test_cap_limits_how_much_one_row_can_matter(self):
        # 20,000 ordinary shipments worth $100 each ($2m of total weight) plus
        # one $500m shipment. Without a cap that single row is 99.6% of the
        # weight; with a $1m cap it is a third.
        n = 20_000
        y = np.concatenate([np.full(n, 100.0), [500_000_000.0]])
        p = np.concatenate([np.full(n, 100.0), [100_000_000.0]])  # big row 5x off

        uncapped = value_weighted_log_mae(y, p, cap=float("inf"))
        capped = value_weighted_log_mae(y, p, cap=1_000_000.0)

        # uncapped it is essentially the big row's own error, log(5)
        assert uncapped == pytest.approx(np.log(5.0), rel=0.01)
        # capped, that row carries about a third of the weight
        assert capped == pytest.approx(np.log(5.0) * 1e6 / (1e6 + 100 * n), rel=0.01)
        assert capped < uncapped / 2

    def test_cap_above_every_value_changes_nothing(self):
        y = np.array([1.0, 10.0, 100.0])
        p = np.array([1.1, 9.0, 130.0])

        assert value_weighted_log_mae(y, p, cap=1e12) == pytest.approx(
            value_weighted_log_mae(y, p, cap=float("inf")))


class TestCalibrationByDecile:
    def test_reports_one_row_per_decile(self):
        rng = np.random.default_rng(0)
        y = np.exp(rng.normal(6, 2, 5000))
        p = y * np.exp(rng.normal(0, 0.3, 5000))

        rows = calibration_by_decile(y, p, bins=10)

        assert len(rows) == 10
        assert sum(r["rows"] for r in rows) == 5000

    def test_detects_systematic_shrinkage(self):
        # A model that hedges toward the middle: high values under-predicted,
        # low values over-predicted. This is the signature the scalar metric
        # hides and the table must show.
        rng = np.random.default_rng(1)
        y = np.exp(rng.normal(6, 2, 20000))
        p = np.exp(0.7 * np.log(y) + 0.3 * 6.0)      # shrunk toward the mean
        rows = calibration_by_decile(y, p, bins=10)

        assert rows[0]["ratio"] > 1.0        # smallest decile over-predicted
        assert rows[-1]["ratio"] < 1.0       # largest decile under-predicted

    def test_a_calibrated_model_has_ratios_near_one(self):
        rng = np.random.default_rng(2)
        y = np.exp(rng.normal(6, 2, 20000))
        p = y * np.exp(rng.normal(0, 0.2, 20000))

        rows = calibration_by_decile(y, p, bins=5)

        for r in rows:
            assert 0.8 < r["ratio"] < 1.25


def build(call, properties=None):
    """A PostTest with a recording cleaner and loader, for population tests."""
    props = {"reference": "sample_enum_transformer", "path": "dataset", "file": "cfs_2017"}
    props.update(properties or {})
    cleaner = PassthroughCleaner()
    pt = PostTest.parse_instruction(props, call, cleaner, None)
    pt.loader = RecordingLoader()
    return pt, cleaner


class TestEvaluationPopulation:
    """Scoring population is decoupled from the training population.

    A model trained with outliers or whole segments removed still has to be
    *scored* on the full population, or a trimmed model can never be compared
    with an untrimmed one. Before this, post_test reused the training cleaner's
    row filters, so trimming the training data silently trimmed the score too --
    the model would look better precisely because it was graded on the easy rows
    it had been allowed to ignore.
    """

    CALL = [{"n_samples": 100, "metrics": ["mae"], "seed": 1}]

    def test_defaults_to_the_training_population(self):
        pt, _ = build(self.CALL)

        assert pt.population == PostTest.TRAINING_POPULATION

    def test_training_population_applies_row_filters(self):
        pt, cleaner = build(self.CALL, {"population": "training"})

        pt.pick_random_samples()

        assert cleaner.calls == [
            {"apply_population_filters": True, "record_metadata": False}
        ]

    def test_all_population_skips_row_filters(self):
        pt, cleaner = build(self.CALL, {"population": "all"})

        pt.pick_random_samples()

        assert cleaner.calls == [
            {"apply_population_filters": False, "record_metadata": False}
        ]

    def test_evaluation_never_records_training_metadata(self):
        # post_test reuses the run's cleaner, so recording here would overwrite
        # the time_ms.cleaning and size.clean.* figures from the training pass.
        for population in ("training", "all"):
            pt, cleaner = build(self.CALL, {"population": population})
            pt.pick_random_samples()
            assert cleaner.calls[0]["record_metadata"] is False

    def test_unknown_population_is_rejected(self):
        with pytest.raises(ValueError, match="population"):
            build(self.CALL, {"population": "everything"})
