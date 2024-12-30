import pandas as pd
import pickle
import numpy as np
import mlflow
import os

from copy import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature

from train.column import CommodityFlow
from train.sstruct import Stage, TrainTestPair, Pairs
from train.scenario_manager import ScenarioManager
import train.data_loader as data_loader
import train.data_cleaner as data_cleaner
        
class Train:
    def __init__(self):
        self.train_pair = None
        self.valid_pair = None
        self.test_pair = None

        self.is_using_tracker = False
        self.tracker_path = None
        self.experiment_name = None

    def set_run_name(self, name):
        self.run_name = name
        return self

    def set_tracker(self, is_using_tracker):
        self.is_using_tracker = is_using_tracker
        return self

    def set_train_name(self, name):
        self.experiment_name = name
        return self

    def train_data(self, pairs):
        self.train_pair = pairs.get_train_pair()
        self.valid_pair = pairs.get_valid_pair()
        self.test_pair = pairs.get_test_pair()
        if self.is_using_tracker:
            return self.train_data_with_tracker()
        return self

    def train_data_with_tracker(self):
        params, lr = self.basic_linear_regression()
        train_mse, valid_mse = self.train_validate(params, lr)

        mlflow.log_params(params)
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("valid_mse", valid_mse)
        mlflow.log_metric("total_trained", self.train_pair.X.shape[0])
        mlflow.log_metric("total_features", self.train_pair.X.shape[1])
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="cfs_model",
            signature=infer_signature(self.train_pair.x_array(), lr.predict(self.train_pair.x_array())),
            input_example=self.train_pair.x_array()[:5],
            registered_model_name="hello",
        )
        return self

    def train_validate(self, param, learn):
        learn.fit(self.train_pair.x_array(), self.train_pair.y)

        y_pred_train = learn.predict(self.train_pair.x_array())
        train_mse = mean_squared_error(self.train_pair.y, y_pred_train)

        y_pred_valid = learn.predict(self.valid_pair.x_array())
        valid_mse = mean_squared_error(self.valid_pair.y, y_pred_valid)
        return train_mse, valid_mse

    def basic_linear_regression(self):
        return {}, LinearRegression()

DATASET_PATH = "dataset/cfs_2017.csv"
TRACKER_PATH = "http://mlflow:5000" # see docker compose for details
RUN_NAME = "base_run"

def train():
    dataloader = data_loader.Disk(DATASET_PATH, CommodityFlow)
    datacleaner = data_cleaner.DataFrame()
    datatransform = DataTransform(CommodityFlow)
    train = (Train()
        .set_train_name("humamtest"))
    (ScenarioManager()
        .set_run_name(RUN_NAME)
        .set_tracking(TRACKER_PATH, "humamtest")
        .start_run("long_run")
        .set_dataloader(dataloader)
        .set_datacleaner(datacleaner)
        .set_datatransform(datatransform)
        .set_train(train)
        .default_path()
        .end_run()
     )

if __name__ == "__main__":
    train()
