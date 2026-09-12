import pandas as pd
from typing import List
import numpy as np

from sklearn.metrics import mean_squared_error
from repositories.repo import Facade
from repositories.struct import TransformationMethods
from .data_io import Disk
from .data_cleaner import Cleaner
from dataclasses import dataclass
from sklearn import metrics as mm
# TODO: should be generalized for all tabular column
from column.cfs2017 import TabularColumn

DEFAULT_N_SAMPLES = 1000
DEFAULT_SEED = 42

# Dollar error is dominated by a handful of enormous shipments, so it does not
# converge: the same model varies ~34% across 100k draws of CFS 2017, and a
# larger draw is worse rather than better. Weighting the *relative* error by
# value keeps the dollar alignment -- dollar error is approximately
# value x relative error -- while capping how much the extreme tail can swing
# the result. That brings the spread to ~6%.
#
# USD 1,000,000 is roughly the 99.9th percentile of SHIPMT_VALUE: 0.086% of rows
# exceed it, and they hold 46% of all value.
VWLE_CAP = 1_000_000.0

_FLOOR = 1e-9


def value_weighted_log_mae(y_true, y_pred, cap: float = VWLE_CAP) -> float:
    """Value-weighted absolute log error, with the weight capped.

        sum( min(y, cap) * |log(y_pred / y)| ) / sum( min(y, cap) )

    A stable, dollar-aligned alternative to raw dollar MAE. Values at or below
    the cap keep their natural weight; everything above is flattened to the cap
    so a single nine-figure shipment cannot own the metric.
    """
    y = np.clip(np.asarray(y_true, dtype=float).reshape(-1), _FLOOR, None)
    p = np.clip(np.asarray(y_pred, dtype=float).reshape(-1), _FLOOR, None)
    weight = np.minimum(y, cap)
    return float((weight * np.abs(np.log(p / y))).sum() / weight.sum())


def calibration_by_decile(y_true, y_pred, bins: int = 10):
    """Mean actual vs mean predicted per value decile.

    A scalar metric can improve while the model keeps shrinking extremes toward
    the middle. This table is what shows whether it actually stopped.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    edges = np.quantile(y, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (y >= lo) & (y <= hi if i == len(edges) - 2 else y < hi)
        if m.sum() == 0:
            continue
        rows.append({
            "decile": i,
            "value_min": float(y[m].min()),
            "value_max": float(y[m].max()),
            "rows": int(m.sum()),
            "log_mae": float(np.abs(np.log(np.clip(p[m], _FLOOR, None) /
                                            np.clip(y[m], _FLOOR, None))).mean()),
            "mean_actual": float(y[m].mean()),
            "mean_predicted": float(p[m].mean()),
            "ratio": float(p[m].mean() / y[m].mean()) if y[m].mean() else float("nan"),
        })
    return rows

# `check_against` selects what a prediction is compared against. Only comparing
# against the row's own actual target is implemented; "random" was the value
# the existing configs used to describe their *sampling*, so it is accepted as
# an alias rather than silently ignored.
_SUPPORTED_CHECK_AGAINST = ("actual_value", "random")


def _resolve(call_value, properties_value, default):
    """A step's `call` wins over its `properties`, then the default.

    Reading the sample size from `properties` alone was a silent trap: no
    config in train_config/ sets `properties.n_rows`, so every post_test
    measured 1000 rows while recording `n_samples` from the call -- 100000 --
    in `size.post_test.row`.
    """
    if call_value is not None:
        return int(call_value)
    if properties_value is not None:
        return int(properties_value)
    return default


@dataclass
class Config:
    intent: str
    check_against: str
    n_samples: int
    metrics: List[str]
    seed: int

@dataclass
class Transformation:
    available_input: List[str]
    transformation: object
    itransformation: object

@dataclass
class Inference:
    transformations: Transformation
    model: callable


class PostTest:
    # Which rows the score is computed over.
    #   "training" - reuse the training cleaner's row filters, so a segment or
    #                trimmed model is scored on its own population
    #   "all"      - skip those filters and score on everything, so models
    #                trained on different populations are still comparable
    TRAINING_POPULATION = "training"
    ALL_POPULATION = "all"
    _POPULATIONS = (TRAINING_POPULATION, ALL_POPULATION)

    # TODO: should support multiple cleaner
    def __init__(self, 
        configs: List[Config], 
        facade: Facade, 
        loader: Disk, 
        cleaner: Cleaner, 
        column_reference: TabularColumn,
        population: str = TRAINING_POPULATION):
        self.facade = facade
        self.loader = loader
        self.configs = configs
        self.cleaner = cleaner
        self.column = column_reference
        self.population = population
        pass

    @classmethod
    def parse_instruction(cls, properties: dict, call: List[dict],cleaner: Cleaner,  facade: Facade ):
        configs = []
        for c in call:
            check_against = c.get("check_against", "actual_value")
            if check_against not in _SUPPORTED_CHECK_AGAINST:
                raise ValueError(
                    f"post_test check_against {check_against!r} is not supported; "
                    f"expected one of {list(_SUPPORTED_CHECK_AGAINST)}"
                )
            configs.append(Config(
                intent=c.get("intent", ""),
                check_against=check_against,
                n_samples=_resolve(c.get("n_samples"), properties.get("n_rows"), DEFAULT_N_SAMPLES),
                metrics=c.get("metrics", ["mse", "rmse"]),
                seed=_resolve(c.get("seed"), properties.get("random_state"), DEFAULT_SEED),
            ))
        if not configs:
            raise ValueError("post_test requires at least one entry in its call list")

        population = properties.get("population", cls.TRAINING_POPULATION)
        if population not in cls._POPULATIONS:
            raise ValueError(
                f"post_test population {population!r} is not supported; "
                f"expected one of {list(cls._POPULATIONS)}"
            )

        column = TabularColumn.from_string(properties.get("reference"))
        loader = Disk(facade, properties.get("path", ""), properties.get("file", ""))
        return cls(
            configs=configs,
            facade=facade,
            loader=loader,
            cleaner=cleaner,
            column_reference=column,
            population=population,
        )

    def reconstruct_inference(self, run_id: str):
        '''
        Reconstruct the inference machine from the saved run
        Similar to the construct method in server/transformation.py and server/model.py

        :param self: self
        :param run_id: main run id of the desired inference machine
        :type run_id: str
        '''
        instructions = self.facade.load_transformation_instruction(run_id)
        transformations, itransformations = [], []
        for step in instructions:
            fobject = self.facade.load_transformation_object(run_id, step.id, step.type)
            if step.inverse_transform:
                func = getattr(fobject.object, "inverse_transform", None)
                if callable(func):
                    itransformations.append({
                        "name": step.name,
                        "column": step.column,
                        "method": step.method,
                        "function": func
                    })
            else:
                func = getattr(fobject.object, "transform", None)
                get_feature_names_out = getattr(fobject.object, "get_feature_names_out", None)
                if callable(func):
                    transformations.append({
                        "name": step.name,
                        "column": step.column,
                        "method": step.method,
                        "function": func,
                        "feature_names": get_feature_names_out,
                    })
        transformation = Transformation(
            available_input=self.facade.get_available_input(run_id),
            transformation=transformations,
            itransformation=itransformations
        )

        model = self.facade.get_model_best_model(run_id).object
        return Inference(transformations=transformation, model=model)

    def pick_random_samples(self, n_samples: int = None, seed: int = None) -> pd.DataFrame:
        """Draw a random sample and run it through the run's cleaner.

        The sample size and seed come from the post_test step's `call`, so the
        metric reflects what the config asked for. They used to be taken from
        `properties`, which no config sets, so every measurement silently fell
        back to the loader default of 1000 rows.
        """
        if n_samples is None:
            n_samples = self.configs[0].n_samples
        if seed is None:
            seed = self.configs[0].seed
        self.loader.load_random_rows_via_csv(
            column=self.column,
            n_rows=n_samples,
            random_state=seed,
            load_options={},
        )
        # record_metadata=False: this is a scoring pass over the run's own
        # cleaner, and writing time_ms.cleaning / size.clean.* here would
        # overwrite the figures the training pass just recorded.
        return self.cleaner.clean_data(
            self.loader.execute(None),
            apply_population_filters=(self.population == self.TRAINING_POPULATION),
            record_metadata=False,
        )

    def predict(self, inference: Inference, samples: pd.DataFrame) -> np.ndarray:
        """Replay the stored transformation pipeline and predict, in dollars.

        Split out of `check` so callers can get the raw predictions for
        diagnostics (calibration by decile, error decomposition) rather than
        only a scalar.
        """
        transformed = samples[[ai.name for ai in inference.transformations.available_input]].copy()
        for transformation in inference.transformations.transformation:
            column = transformation["column"]
            if column in self.column.categorical():
                input = transformed[column].astype(str)
            else:
                input = transformed[column].astype(float)
            method = transformation.get("method", "")
            if method == TransformationMethods.REPLACE.name:
                # TODO: need to do something better than this to convert the type
                transformed[column] = transformed[column].astype(float)
                transformed.loc[:, column] = transformation["function"](input.to_numpy().reshape(-1, 1))
            elif method == TransformationMethods.APPEND.name:
                appended = transformation["function"](input.to_numpy().reshape(-1, 1))
                transformed[f"{column}_{transformation['name']}"] = appended
            elif method == TransformationMethods.APPEND_AND_REMOVE.name:
                appended = transformation["function"](input.to_numpy().reshape(-1, 1))
                encoded_columns = transformation["feature_names"]([column])
                new_columns = pd.DataFrame(appended, columns=encoded_columns, dtype='int', index=transformed.index)
                transformed = pd.concat([transformed.drop(column, axis=1), new_columns], axis=1)
        result = inference.model.predict(transformed)
        for it in inference.transformations.itransformation:
            # the target's inverse transform, e.g. exp, maps back to dollars
            result = it["function"](np.asarray(result).reshape(-1, 1))
        return np.asarray(result).reshape(-1)

    def check(self, inference: Inference, samples: pd.DataFrame, metrics: List[str]) -> dict:
        actual_outcome = samples[self.column.target()].to_numpy(dtype=float)
        predicted = self.predict(inference, samples)
        return {
            metric: self.metric_map()[metric](actual_outcome, predicted)
            for metric in metrics
        }

    def metric_map(self):
        return {
            "mse": mean_squared_error,
            "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
            "value_weighted_log_mae": value_weighted_log_mae,
        }

    def store_metrics(self, result: dict):
        for metric, value in result.items():
            print(f"Post Test Metric {metric}: {value}")
            self.facade.set_metric("post_test", metric, value)

    def execute(self, _ : any):
        '''
        execute callable is the standard scenario manager interface
        any defined module in the traning configs should have this function
        so scenario manager could execute and call the intended function
        '''
        run_id = self.facade.current_run_id
        if run_id == "":
            raise ValueError("Post Test require reference run id. Should be main run_id")

        for pt in self.configs:
            self.facade.set_post_test_row_size(run_id, pt.n_samples)
            self.facade.set_post_test_intent(run_id, pt.intent)

            inference_machine = self.reconstruct_inference(run_id)
            samples = self.pick_random_samples(pt.n_samples, pt.seed)
            print(f"Post Test: {len(samples)} rows sampled from {pt.n_samples} drawn (seed {pt.seed})")
            result = self.check(inference_machine, samples, pt.metrics)
            self.store_metrics(result)

        # return the original object
        return object

