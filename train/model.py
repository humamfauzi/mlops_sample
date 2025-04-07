import mlflow
import random

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature

from enum import Enum
from typing import Optional
from train.sstruct import Pairs, FeatureTargetPair

class TabularModel(ABC):
    @abstractmethod
    def train_data(self, pairs: Pairs):
        return self
    
    @abstractmethod
    def set_run_name(self, name: str):
        return self
        
class ModelScenario(Enum):
    BASIC = 1
    MULTI_REGRESSION = 2

# TODO find out whether separating each model to its own class is a good idead
# the premise is that we only load the kind of model we want to train to
# parent class which is this
class Model(TabularModel):
    def __init__(self):
        self.pairs = Optional[Pairs]

        self.is_using_tracker = False
        self.tracker_path = None

        self.scenario: Optional[ModelScenario] = None

    def set_run_name(self, name):
        self.run_name = name
        return self

    def set_tracker(self, is_using_tracker):
        self.is_using_tracker = is_using_tracker
        return self

    def set_scenario(self, scenario_name: ModelScenario):
        self.scenario = scenario_name
        return self

    # the objective with train data is that
    # it would send metrics to mlflow and save artifacts there
    def train_data(self, pairs: Pairs) -> 'Model':
        if self.scenario == ModelScenario.MULTI_REGRESSION:
            return self.train_data_with_multiple_regression()
        else:
            return self.train_data_with_tracker()

    def generate_random_string(self, length: int) -> str:
        char = "ABCDEFGHIJKLMOPQRSTUVWXYZ12345678890"
        final = ""
        for _ in range(length):
            final += random.choice(char)
        return final

    def train_data_with_multiple_regression(self) -> 'Model' :
        if self.train_pair is None:
            raise ValueError("Train Pair is none; required one to proceed")
        lr = LinearRegression()
        en = ElasticNet()
        la = Lasso()
        dt = DecisionTreeRegressor()
        rf = RandomForestRegressor()
        gb = GradientBoostingRegressor()
        knn = KNeighborsRegressor()
        models = [lr, en, la, dt, rf, gb, knn]
        prefix = self.generate_random_string(4)
        for model in models:
            model_name = model.__class__.__name__
            train_mse, valid_mse = self.train_validate(model)
            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("valid_mse", valid_mse)
            mlflow.log_metric("total_trained", self.train_pair.X.shape[0])
            mlflow.log_metric("total_features", self.train_pair.X.shape[1])
            mlflow.set_tag("level", "candidate")
            mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path=f"cfs_mode/{model_name}",
                signature=infer_signature(self.train_pair.x_array(), lr.predict(self.train_pair.x_array())),
                input_example=self.train_pair.x_array()[:5],
                registered_model_name=prefix+"_"+model_name
            )
        return self

    # TODO handle multiple model and hyper parameter gridsearch
    # TODO auto tagging the best model
    # TODO use the basic training as base metrics so we can compare it in dashboard
    # TODO the features seems not saved. Need to find out how to save feature columns
    # TODO testing this function requires mock all mlflow methods
    def train_data_with_tracker(self):
        if self.train_pair is None:
            raise ValueError("Train Pair is none; required one to proceed")
        params, lr = self.basic_linear_regression()
        train_mse, valid_mse = self.train_validate(lr)

        mlflow.log_params(params)
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("valid_mse", valid_mse)
        mlflow.log_metric("total_trained", self.train_pair.X.shape[0])
        mlflow.log_metric("total_features", self.train_pair.X.shape[1])
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="cfs_model",
            signature=infer_signature(self.train_pair.x_array(), lr.predict(self.train_pair.x_array())),
            input_example=self.train_pair.x_array()[:5],
            registered_model_name="hello",
        )
        return self


    # TODO generalize it so it can handle train, valid, and test well
    def train_validate(self, learn):
        if self.train_pair is None:
            raise ValueError("Train pair required for validating")
        if self.valid_pair is None:
            raise ValueError("Train pair required for validating")
        learn.fit(self.train_pair.x_array(), self.train_pair.y)

        y_pred_train = learn.predict(self.train_pair.x_array())
        train_mse = mean_squared_error(self.train_pair.y, y_pred_train)

        y_pred_valid = learn.predict(self.valid_pair.x_array())
        valid_mse = mean_squared_error(self.valid_pair.y, y_pred_valid)
        return train_mse, valid_mse

    def basic_linear_regression(self):
        return {}, LinearRegression()
