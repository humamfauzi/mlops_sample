import random
import pprint
from time import time
import numpy as np

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

from enum import Enum
from typing import Optional, List
from train.sstruct import Pairs
from sklearn.model_selection import ParameterGrid
from repositories.struct import ModelObject
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
        self.facade = facade
        self.run_id = run_id

    def train(self, pairs: Pairs):
        start = time.time()
        if not isinstance(pairs, Pairs):
            raise TypeError("Input data must be of type Pairs")
        self.model.fit(pairs.train.x_array(), np.array(pairs.train.y).reshape(-1,))
        end = time.time()
        duration_ms = (end - start) * 1000.0
        self.facade.set_training_time(duration_ms)
        return self

    def validate(self, pairs: Pairs, metrics: List[str]):
        if not isinstance(pairs, Pairs):
            raise TypeError(f"Input data must be of type Pairs but get {type(pairs)}")
        y_pred = self.model.predict(pairs.valid.x_array())
        mm = self.metric_map()
        pairs = {"train": pairs.train, "valid": pairs.valid}
        for metric in metrics:
            for stage in ["train", "valid"]:
                start = time.time()
                if metric in mm:
                    y_pred = self.model.predict(pairs[stage].x_array())
                    value = mm[metric](pairs[stage].y, y_pred)
                    duration_ms = (time.time() - start) * 1000.0
                    self.facade.set_validation_time(stage, duration_ms)
                    self.facade.set_metric(stage, metric, value)
        return self

    def metric_map(self):
        return {
            "mse": mean_squared_error,
            "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
        }

    def test(self, pairs: Pairs, metrics: List[str]):
        if not isinstance(pairs, Pairs):
            raise TypeError(f"Input data must be of type Pairs but get {type(pairs)}")
        start = time.time()
        y_pred = self.model.predict(pairs.test.x_array())
        mm = self.metric_map()
        for metric in metrics:
            if metric in mm:
                value = mm[metric](pairs.test.y, y_pred)
                self.facade.set_metric("test", metric, value)
        self.facade.set_validation_time("test", (time.time() - start) * 1000.0)
        return self

    def set_as_the_best(self):
        self.facade.tag_as_the_best()
        return self

    def save(self):
        model_object = ModelObject(filename=f"{self.run_id}-{self.name}", object=self.model)
        self.facade.save_model(model_object)
        self.facade.set_model_properties(self.hyperparameters)
        return self

class ModelTrainer:
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
            metrics=[]
        ):
        self.facade = facade
        self.random_state = random_state
        self.objective = objective
        self.parameter_grid = parameter_grid
        self.fold = fold
        self.metrics = metrics
        self.models = []
        pass

    @classmethod
    def parse_instruction(cls, properties: dict, call: List[dict], facade):
        m = cls(facade, **properties)
        for step in call:
            m.add_model(step)
        return m

    def execute(self, input_data: Pairs) -> None:
        if not isinstance(input_data, Pairs):
            raise TypeError("Input data must be of type Pairs")
        for model in self.models:
            self.facade.new_child_run(model.run_id)
            model.train(input_data)
            model.validate(input_data, self.metrics)
            model.save()

        best_model = self.compare_model()
        self.check_model_against_test(best_model, input_data)
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
        elif model_type == "elastic_net":
            return ElasticNet
        elif model_type == "lasso":
            return Lasso
        elif model_type == "k_neighbors_regressor":
            return KNeighborsRegressor
        else:
            raise ValueError(f"Model type {model_type} is not supported")

    def generate_model(self, model_type, hyperparameters, grid_type):
        models = []
        model_class = self.model_routing(model_type)
        if grid_type == "random":
            # TODO implement random grid search
            pass
        else:
            param_grid = ParameterGrid(hyperparameters)
            for pg in list(param_grid):
                model = model_class(**pg)
                mw = ModelWrapper(model_type, model, pg, self.facade.generate_run_id(), self.facade)
                models.append(mw)
        return models

    def compare_model(self) -> ModelWrapper:
        if len(self.models) == 0:
            raise ValueError("No model to compare")
        if self.objective == self.objective_best_model:
            id = self.facade.find_best_model("rmse")
            best_model = self.find_model_by_id(id)
            best_model.set_as_the_best()
            return best_model
        else:
            # TODO implement fast model selection
            pass
        return self.models[0]

    def find_model_by_id(self, id: str) -> Optional[ModelWrapper]:
        for model in self.models:
            if model.run_id == id:
                return model
        raise ValueError(f"Model with id {id} not found")

    def check_model_against_test(self, best_model: ModelWrapper, input_data: Pairs):
        if best_model is None:
            raise ValueError("Best model is None")
        best_model.test(input_data, self.metrics)
        return self