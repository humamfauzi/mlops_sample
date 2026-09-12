"""Tests for model validation, test scoring and trainer option handling.

Covers F-05 and F-16 from REPOSITORY_MAP.md, plus the F-14 options that used to
be accepted and silently ignored.
"""
import numpy as np
import pandas as pd
import pytest

from train.model import ModelTrainer, ModelWrapper
from train.sstruct import FeatureTargetPair, Pairs, Stage


class RecordingFacade:
    """Captures the metrics and timings a trainer would have recorded."""

    def __init__(self):
        self.metrics = {}
        self.times = {}
        self.tagged_best = []
        self._next_id = 0

    def set_metric(self, stage, metric, value):
        self.metrics[(stage, metric)] = value

    def set_validation_time(self, stage, ms):
        self.times[stage] = ms

    def set_training_time(self, ms):
        self.training_time = ms

    def set_training_type(self, name):
        self.training_type = name

    def tag_as_the_best(self):
        self.tagged_best.append(True)

    def generate_run_id(self):
        self._next_id += 1
        return f"RUN{self._next_id:03d}"


class StubModel:
    """Predicts a constant, and counts how often it was asked to."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        return np.full(len(X), self.value)


def make_pairs(y_true=(0.0, 0.0, 9.0)):
    """A Pairs whose three splits all carry the same small target."""
    y = pd.DataFrame({"target": list(y_true)})
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    return Pairs(
        train=FeatureTargetPair(X.copy(), y.copy(), Stage.TRAIN),
        valid=FeatureTargetPair(X.copy(), y.copy(), Stage.VALID),
        test=FeatureTargetPair(X.copy(), y.copy(), Stage.TEST),
    )


class TestTestScoring:
    """F-05: test() returned the loop variable, not the primary metric."""

    # y_true = [0, 0, 9] with a constant prediction of 3 gives deliberately
    # different values for each metric, so they cannot be confused.
    #   mae  = (3 + 3 + 6) / 3      = 4.0
    #   mse  = (9 + 9 + 36) / 3     = 18.0
    #   rmse = sqrt(18)             ~= 4.2426
    MAE, MSE, RMSE = 4.0, 18.0, np.sqrt(18.0)

    def test_returns_every_requested_metric(self):
        facade = RecordingFacade()
        wrapper = ModelWrapper("stub", StubModel(3.0), {}, "RUN001", facade)

        scores = wrapper.test(make_pairs(), ["mae", "mse", "rmse"])

        assert scores["mae"] == pytest.approx(self.MAE)
        assert scores["mse"] == pytest.approx(self.MSE)
        assert scores["rmse"] == pytest.approx(self.RMSE)

    def test_records_every_metric_on_the_test_stage(self):
        facade = RecordingFacade()
        wrapper = ModelWrapper("stub", StubModel(3.0), {}, "RUN001", facade)

        wrapper.test(make_pairs(), ["mae", "mse", "rmse"])

        assert set(facade.metrics) == {
            ("test", "mae"), ("test", "mse"), ("test", "rmse")
        }

    def test_check_against_test_returns_the_primary_metric_not_the_last(self):
        # The regression: metrics ended with "rmse" but the primary metric was
        # "mae". Nomination compared the returned RMSE against the incumbent's
        # stored MAE.
        facade = RecordingFacade()
        trainer = ModelTrainer(facade, metrics=["mae", "rmse"], primary_metric="mae")
        wrapper = ModelWrapper("stub", StubModel(3.0), {}, "RUN001", facade)

        score = trainer.check_model_against_test(wrapper, make_pairs())

        assert score == pytest.approx(self.MAE)
        assert score != pytest.approx(self.RMSE)

    def test_primary_metric_is_honoured_whatever_the_order(self):
        facade = RecordingFacade()
        trainer = ModelTrainer(facade, metrics=["rmse", "mae"], primary_metric="rmse")
        wrapper = ModelWrapper("stub", StubModel(3.0), {}, "RUN001", facade)

        assert trainer.check_model_against_test(wrapper, make_pairs()) == pytest.approx(self.RMSE)

    def test_missing_primary_metric_is_reported(self):
        facade = RecordingFacade()
        trainer = ModelTrainer(facade, metrics=["mae"], primary_metric="mae")
        wrapper = ModelWrapper("stub", StubModel(3.0), {}, "RUN001", facade)

        # ask for a metric the wrapper is not configured to compute
        trainer.metrics = ["mse"]

        with pytest.raises(ValueError, match="was not computed"):
            trainer.check_model_against_test(wrapper, make_pairs())

    def test_predicts_only_once(self):
        facade = RecordingFacade()
        model = StubModel(3.0)
        wrapper = ModelWrapper("stub", model, {}, "RUN001", facade)

        wrapper.test(make_pairs(), ["mae", "mse", "rmse"])

        assert model.calls == 1


class TestValidation:
    """F-16: validate() re-predicted for every metric *and* every stage."""

    def test_predicts_once_per_split(self):
        facade = RecordingFacade()
        model = StubModel(3.0)
        wrapper = ModelWrapper("stub", model, {}, "RUN001", facade)

        wrapper.validate(make_pairs(), ["mae", "mse", "rmse"])

        # three metrics over two splits used to run six predictions
        assert model.calls == 2

    def test_records_metrics_for_both_stages(self):
        facade = RecordingFacade()
        wrapper = ModelWrapper("stub", StubModel(3.0), {}, "RUN001", facade)

        wrapper.validate(make_pairs(), ["mae", "mse", "rmse"])

        assert {stage for stage, _ in facade.metrics} == {"train", "valid"}
        assert len(facade.metrics) == 6

    def test_records_one_timing_per_stage(self):
        facade = RecordingFacade()
        wrapper = ModelWrapper("stub", StubModel(3.0), {}, "RUN001", facade)

        wrapper.validate(make_pairs(), ["mae", "mse", "rmse"])

        assert set(facade.times) == {"train", "valid"}

    def test_metric_values_are_unchanged(self):
        facade = RecordingFacade()
        wrapper = ModelWrapper("stub", StubModel(3.0), {}, "RUN001", facade)

        wrapper.validate(make_pairs(), ["mae", "mse"])

        assert facade.metrics[("valid", "mae")] == pytest.approx(4.0)
        assert facade.metrics[("valid", "mse")] == pytest.approx(18.0)


class TestUnimplementedOptions:
    """F-14: options that were accepted and then quietly ignored."""

    def test_random_parameter_grid_is_rejected(self):
        facade = RecordingFacade()
        trainer = ModelTrainer(facade, metrics=["mae"], primary_metric="mae")

        # "random" used to be a no-op that fell through to the exhaustive grid
        with pytest.raises(ValueError, match="not implemented"):
            trainer.generate_model("linear_regression", {}, "random")

    def test_unknown_parameter_grid_is_rejected(self):
        facade = RecordingFacade()
        trainer = ModelTrainer(facade, metrics=["mae"], primary_metric="mae")

        with pytest.raises(ValueError, match="Unknown parameter_grid"):
            trainer.generate_model("linear_regression", {}, "bayesian")

    def test_exhaustive_grid_still_works(self):
        facade = RecordingFacade()
        trainer = ModelTrainer(facade, metrics=["mae"], primary_metric="mae")

        models = trainer.generate_model(
            "linear_regression", {"fit_intercept": [True, False]}, "exhaustive"
        )

        assert len(models) == 2

    def test_unimplemented_objective_is_rejected(self):
        facade = RecordingFacade()
        trainer = ModelTrainer(
            facade, objective="fast_model", metrics=["mae"], primary_metric="mae"
        )
        trainer.models = [ModelWrapper("stub", StubModel(1.0), {}, "RUN001", facade)]

        with pytest.raises(ValueError, match="not implemented"):
            trainer.compare_model()

    def test_first_model_objective_is_supported(self):
        facade = RecordingFacade()
        trainer = ModelTrainer(
            facade, objective="first_model", metrics=["mae"], primary_metric="mae"
        )
        first = ModelWrapper("stub", StubModel(1.0), {}, "RUN001", facade)
        trainer.models = [first, ModelWrapper("stub", StubModel(2.0), {}, "RUN002", facade)]

        assert trainer.compare_model() is first

    def test_unknown_model_type_is_rejected(self):
        facade = RecordingFacade()
        trainer = ModelTrainer(facade, metrics=["mae"], primary_metric="mae")

        with pytest.raises(ValueError, match="not supported"):
            trainer.generate_model("transformer_network", {}, "exhaustive")

    def test_unsupported_data_io_format_is_rejected(self):
        from train.data_io import Disk

        with pytest.raises(ValueError, match="not supported"):
            Disk.parse_instruction(
                {"path": "dataset", "file": "x", "format": "parquet",
                 "reference": "sample_enum_transformer"},
                [{"type": "load"}],
                None,
            )

    def test_unknown_data_io_step_is_rejected(self):
        from train.data_io import Disk

        with pytest.raises(ValueError, match="Unknown data_io step"):
            Disk.parse_instruction(
                {"path": "dataset", "file": "x", "format": "csv",
                 "reference": "sample_enum_transformer"},
                [{"type": "stream_forever"}],
                None,
            )

    def test_one_hot_encoding_rejects_a_condition_it_ignores(self):
        from train.data_transform import Transformer

        with pytest.raises(ValueError, match="one_hot_encoding only supports"):
            Transformer.parse_instruction(
                {"reference": "sample_enum_transformer"},
                [{"type": "one_hot_encoding", "condition": "replace",
                  "columns": ["column_categorical"]}],
                None,
            )

    def test_one_hot_encoding_accepts_append_and_remove(self):
        from train.data_transform import Transformer

        transformer = Transformer.parse_instruction(
            {"reference": "sample_enum_transformer"},
            [{"type": "one_hot_encoding", "condition": "append_and_remove",
              "columns": ["column_categorical"]}],
            None,
        )

        assert len(transformer.keepers) == 1


class TestHistGradientBoosting:
    """Phase B: the histogram GBM is what makes a grid affordable.

    Classic GradientBoostingRegressor took 173s for a single 200-tree/depth-5
    fit on 559k rows, which is why the published champion is capacity-starved.
    """

    def test_routing_returns_the_histogram_estimator(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        facade = RecordingFacade()
        trainer = ModelTrainer(facade, metrics=["mae"], primary_metric="mae")

        assert trainer.model_routing("hist_gradient_boosting_regressor") is (
            HistGradientBoostingRegressor
        )

    def test_documented_hyperparameters_are_accepted(self):
        # The names differ from the classic GBM: max_iter, not n_estimators;
        # no subsample. Getting this wrong fails at fit time, not parse time.
        facade = RecordingFacade()
        trainer = ModelTrainer(facade, metrics=["mae"], primary_metric="mae")

        models = trainer.generate_model(
            "hist_gradient_boosting_regressor",
            {
                "loss": ["absolute_error"],
                "max_iter": [50],
                "learning_rate": [0.1],
                "max_leaf_nodes": [31],
                "min_samples_leaf": [20],
                "early_stopping": [False],
                "random_state": [42],
            },
            "exhaustive",
        )

        assert len(models) == 1

    def test_rejects_the_classic_gbm_spelling(self):
        # A config written for the classic GBM must fail loudly rather than
        # silently train something unexpected.
        facade = RecordingFacade()
        trainer = ModelTrainer(facade, metrics=["mae"], primary_metric="mae")

        with pytest.raises(TypeError):
            trainer.generate_model(
                "hist_gradient_boosting_regressor",
                {"n_estimators": [100], "subsample": [0.8]},
                "exhaustive",
            )

    def test_trains_and_predicts(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        facade = RecordingFacade()
        wrapper = ModelWrapper(
            "hist_gradient_boosting_regressor",
            HistGradientBoostingRegressor(max_iter=20, early_stopping=False, random_state=0),
            {},
            "RUN001",
            facade,
        )
        pairs = make_pairs()

        wrapper.train(pairs)
        score = wrapper.test(pairs, ["mae"])

        assert facade.training_type == "hist_gradient_boosting_regressor"
        assert score["mae"] >= 0

    def test_grid_expands_over_multiple_values(self):
        facade = RecordingFacade()
        trainer = ModelTrainer(facade, metrics=["mae"], primary_metric="mae")

        models = trainer.generate_model(
            "hist_gradient_boosting_regressor",
            {"loss": ["squared_error", "absolute_error"], "max_iter": [100, 200]},
            "exhaustive",
        )

        assert len(models) == 4
