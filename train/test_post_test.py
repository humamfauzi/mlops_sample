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

import pandas as pd
import pytest

from train.post_test import (
    DEFAULT_N_SAMPLES,
    DEFAULT_SEED,
    Config,
    PostTest,
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
    def execute(self, frame):
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
