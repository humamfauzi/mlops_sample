import random
import pprint
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
    def __init__(self, name, model, hyperparameters):
        self.name = name
        self.model = model
        self.hyperparameters = hyperparameters
        self.metrics = {}
        self.properties = {}
        self.tag = {}
    
    def save_metrics(self, metrics_name:str, metrics_value: float):
        self.metrics[metrics_name] = metrics_value

    def save_properties(self, name: str, value):
        self.properties[name] = value

    def save_tag(self, name: str, value: str):
        self.tag[name] = value

    def train(self, pairs: Pairs):
        if not isinstance(pairs, Pairs):
            raise TypeError("Input data must be of type Pairs")
        self.model.fit(pairs.train.x_array(), np.array(pairs.train.y).reshape(-1,))
        return self

    def validate(self, pairs: Pairs, metrics: List[str]):
        if not isinstance(pairs, Pairs):
            raise TypeError(f"Input data must be of type Pairs but get {type(pairs)}")
        y_pred = self.model.predict(pairs.valid.x_array())
        for metric in metrics:
            if metric == "mse":
                for stage in ["train", "valid"]:
                    if stage == "train":
                        y_pred = self.model.predict(pairs.train.x_array())
                        mse = mean_squared_error(pairs.train.y, y_pred)
                        self.save_metrics("train_mse", mse)
                    else:
                        y_pred = self.model.predict(pairs.valid.x_array())
                        mse = mean_squared_error(pairs.valid.y, y_pred)
                        self.save_metrics("valid_mse", mse)
            elif metric == "rmse":
                for stage in ["train", "valid"]:
                    if stage == "train":
                        y_pred = self.model.predict(pairs.train.x_array())
                        rmse = np.sqrt(mean_squared_error(pairs.train.y, y_pred))
                        self.save_metrics("train_rmse", rmse)
                    else:
                        y_pred = self.model.predict(pairs.valid.x_array())
                        rmse = np.sqrt(mean_squared_error(pairs.valid.y, y_pred))
                        self.save_metrics("valid_rmse", rmse)
        return self

    def test(self, pairs: Pairs, metrics: List[str]):
        if not isinstance(pairs, Pairs):
            raise TypeError(f"Input data must be of type Pairs but get {type(pairs)}")
        y_pred = self.model.predict(pairs.test.x_array())
        for metric in metrics:
            if metric == "mse":
                mse = mean_squared_error(pairs.test.y, y_pred)
                self.save_metrics("test_mse", mse)
            elif metric == "rmse":
                rmse = np.sqrt(mean_squared_error(pairs.test.y, y_pred))
                self.save_metrics("test_rmse", rmse)
        return self

    def log_model(self):
        complete_log = {
            "name": self.name,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "properties": self.properties,
            "tag": self.tag
        }
        pprint.pprint(complete_log)

class ModelTrainer:
    objective_best_model = "best_model"
    objective_fast_model = "fast_model"

    parameter_grid_exhaustive = "exhaustive"
    parameter_grid_random = "random"

    def __init__(self, 
            random_state=42,
            objective="best_model",
            fold=5,
            parameter_grid="exhaustive",
            metrics=[]
        ):
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
            model.train(input_data)
            model.validate(input_data, self.metrics)

        best_model = self.compare_model()
        self.check_model_against_test(best_model, input_data)
        best_model.log_model()
        return 

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
                mw = ModelWrapper(model_type, model, pg)
                models.append(mw)
        return models

    def compare_model(self) -> ModelWrapper:
        if len(self.models) == 0:
            raise ValueError("No model to compare")
        if self.objective == self.objective_best_model:
            best_model = None
            best_metric = float("inf")
            for model in self.models:
                if "valid_mse" in model.metrics:
                    if model.metrics["valid_mse"] < best_metric:
                        best_metric = model.metrics["valid_mse"]
                        best_model = model
                if "valid_rmse" in model.metrics:
                    if model.metrics["valid_rmse"] < best_metric:
                        best_metric = model.metrics["valid_rmse"]
                        best_model = model
            best_model.save_tag("level", "best")
            return best_model
        else:
            # TODO implement fast model selection
            pass
        return self.models[0]

    def check_model_against_test(self, best_model: ModelWrapper, input_data: Pairs):
        if best_model is None:
            raise ValueError("Best model is None")
        best_model.test(input_data, self.metrics)
        return self