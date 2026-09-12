from time import time
import numpy as np

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

from enum import Enum
from typing import Optional, List
from train.sstruct import Pairs
from sklearn.model_selection import ParameterGrid
from repositories.struct import ModelObject
from repositories.repo import Facade
import time

class TabularModel(ABC):
    @abstractmethod
    def train_data(self, pairs: Pairs):
        return self
    
    @abstractmethod
    def set_run_name(self, name: str):
        return self
        
class ModelWrapper:
    """
    A simple wrapper to hold model information
    So all the model result, hyperparameters, and training properties are hold in this class
    not in the trainer
    """
    def __init__(self, name, model, hyperparameters, run_id, facade):
        self.name = name
        self.model = model
        self.hyperparameters = hyperparameters
        self.facade: Facade = facade
        # It is the random string not the integer id
        self.run_id = run_id

    def train(self, pairs: Pairs):
        start = time.time()
        if not isinstance(pairs, Pairs):
            raise TypeError("Input data must be of type Pairs")
        self.model.fit(pairs.train.X, np.array(pairs.train.y).reshape(-1,))
        end = time.time()
        duration_ms = (end - start) * 1000.0
        self.facade.set_training_time(duration_ms)
        self.facade.set_training_type(self.name)
        return self

    def validate(self, pairs: Pairs, metrics: List[str]):
        """Score train and valid splits, predicting once per split.

        The previous version predicted inside the metric loop, so a three-metric
        request ran six full predictions instead of two, and the recorded
        validation time was whichever metric happened to run last.
        """
        if not isinstance(pairs, Pairs):
            raise TypeError(f"Input data must be of type Pairs but get {type(pairs)}")
        mm = self.metric_map()
        for stage, pair in (("train", pairs.train), ("valid", pairs.valid)):
            start = time.time()
            y_pred = self.model.predict(pair.X)
            y_true = np.array(pair.y).reshape(-1)
            for metric in metrics:
                if metric in mm:
                    self.facade.set_metric(stage, metric, mm[metric](y_true, y_pred))
            self.facade.set_validation_time(stage, (time.time() - start) * 1000.0)
        return self

    def metric_map(self):
        return {
            "mse": mean_squared_error,
            "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
        }

    def test(self, pairs: Pairs, metrics: List[str]) -> dict:
        """Score the held-out test split and return every metric by name.

        Returning a mapping rather than the loop variable matters: nomination
        compares the candidate against the incumbent's stored
        `validation.test.<primary_metric>`, so the caller must be able to pick
        the primary metric rather than receive whichever one happened to be
        computed last.
        """
        if not isinstance(pairs, Pairs):
            raise TypeError(f"Input data must be of type Pairs but get {type(pairs)}")
        start = time.time()
        y_pred = self.model.predict(pairs.test.X)
        y_true = np.array(pairs.test.y).reshape(-1)
        mm = self.metric_map()
        scores = {}
        for metric in metrics:
            if metric in mm:
                scores[metric] = mm[metric](y_true, y_pred)
                self.facade.set_metric("test", metric, scores[metric])
        self.facade.set_validation_time("test", (time.time() - start) * 1000.0)
        return scores

    def set_as_the_best(self):
        self.facade.tag_as_the_best()
        return self

    def save(self):
        model_object = ModelObject(filename=f"{self.run_id}", object=self.model)
        self.facade.save_model(model_object)
        self.facade.set_model_properties(self.hyperparameters)
        return self

class ModelTrainer:
    objective_first_model = "first_model" # for test unit purpose
    objective_best_model = "best_model"
    objective_fast_model = "fast_model"

    parameter_grid_exhaustive = "exhaustive"
    parameter_grid_random = "random"

    def __init__(self, 
            facade,
            random_state=42,
            objective="best_model",
            fold=5,
            parameter_grid="exhaustive",
            metrics=[],
            primary_metric=""
        ):
        self.facade: Facade = facade
        self.random_state = random_state
        self.objective = objective
        self.parameter_grid = parameter_grid
        self.fold = fold
        if primary_metric == "" or primary_metric not in metrics:
            raise ValueError("Primary metric must be one of the metrics and not empty")
        self.metrics = metrics
        self.primary_metric = primary_metric
        self.models = []
        pass

    @classmethod
    def parse_instruction(cls, properties: dict, call: List[dict], facade):
        m = cls(facade, **properties)
        for step in call:
            m.add_model(step)
        return m

    def execute(self, input_data: Pairs) -> None:
        print("Starting model training process...")
        if not isinstance(input_data, Pairs):
            raise TypeError("Input data must be of type Pairs")
        for model in self.models:
            self.facade.new_child_run(model.run_id)
            model.train(input_data)
            model.validate(input_data, self.metrics)
            model.save()

        best_model = self.compare_model()
        test_metric = self.check_model_against_test(best_model, input_data)
        self.nominate_for_publishing(test_metric, best_model.run_id)
        return self

    def add_model(self, call: dict):
        gm = self.generate_model(call["model_type"], call["hyperparameters"], self.parameter_grid["type"])
        self.models.extend(gm)
        return self

    def model_routing(self, model_type):
        if model_type == "random_forest_regressor":
            return RandomForestRegressor
        elif model_type == "linear_regression":
            return LinearRegression
        elif model_type == "decision_tree_regressor":
            return DecisionTreeRegressor
        elif model_type == "gradient_boosting_regressor":
            return GradientBoostingRegressor
        elif model_type == "hist_gradient_boosting_regressor":
            # Histogram-based GBM: roughly two orders of magnitude faster than
            # GradientBoostingRegressor at this scale, with built-in early
            # stopping. Note the hyperparameter names differ -- it takes
            # `max_iter` rather than `n_estimators`, has no `subsample`, and
            # defaults to `min_samples_leaf=20` rather than 1.
            return HistGradientBoostingRegressor
        elif model_type == "elastic_net":
            return ElasticNet
        elif model_type == "lasso":
            return Lasso
        elif model_type == "k_nearest_neighbors_regressor":
            return KNeighborsRegressor
        else:
            raise ValueError(f"Model type {model_type} is not supported")

    def generate_model(self, model_type, hyperparameters, grid_type):
        if grid_type == self.parameter_grid_random:
            # Never implemented. It used to be a no-op that fell through to the
            # exhaustive branch, so asking for a cheap random search silently
            # ran the most expensive path available.
            raise ValueError(
                f"parameter_grid {grid_type!r} is not implemented; "
                f"use {self.parameter_grid_exhaustive!r}"
            )
        if grid_type != self.parameter_grid_exhaustive:
            raise ValueError(
                f"Unknown parameter_grid {grid_type!r}; "
                f"expected {self.parameter_grid_exhaustive!r}"
            )
        models = []
        model_class = self.model_routing(model_type)
        for pg in list(ParameterGrid(hyperparameters)):
            model = model_class(**pg)
            mw = ModelWrapper(model_type, model, pg, self.facade.generate_run_id(), self.facade)
            models.append(mw)
        return models

    def compare_model(self) -> ModelWrapper:
        if len(self.models) == 0:
            raise ValueError("No model to compare")
        if self.objective == self.objective_best_model:
            id = self.facade.find_best_model(self.primary_metric)
            best_model = self.find_model_by_id(id)
            best_model.set_as_the_best()
            return best_model
        if self.objective == self.objective_first_model:
            # Test-only objective: take the first model off the grid.
            self.models[0].set_as_the_best()
            return self.models[0]
        # "fast_model" was declared but never implemented. This branch used to
        # fall through and silently return models[0].
        raise ValueError(
            f"objective {self.objective!r} is not implemented; supported: "
            f"{self.objective_best_model!r}, {self.objective_first_model!r}"
        )

    def find_model_by_id(self, id: str) -> Optional[ModelWrapper]:
        for model in self.models:
            if model.run_id == id:
                return model
        raise ValueError(f"Model with id {id} not found")

    def check_model_against_test(self, best_model: ModelWrapper, input_data: Pairs) -> float:
        """Score the chosen model on test and return its *primary* metric.

        This value becomes the candidate score in `nominate_for_publishing`,
        which is compared against the incumbent's stored
        `validation.test.<primary_metric>`. Returning any other metric would
        compare two different things.
        """
        if best_model is None:
            raise ValueError("Best model is None")
        scores = best_model.test(input_data, self.metrics)
        if self.primary_metric not in scores:
            raise ValueError(
                f"Primary metric {self.primary_metric!r} was not computed; "
                f"got {sorted(scores)}"
            )
        return scores[self.primary_metric]

    def nominate_for_publishing(self, current_score: float, model_id: str):
        intent = self.facade.get_intent()
        _, parent_id = self.facade.get_model_run_id(model_id)
        self.facade.nominate_for_publishing(intent, self.primary_metric, current_score, parent_id)