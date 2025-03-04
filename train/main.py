from train.scenario_manager import PreprocessScenarioManager, ModelScenarioManager
from column.cfs2017 import CommodityFlow
import os
import random
import mlflow

import train.data_loader as data_loader
import train.data_cleaner as data_cleaner
import train.data_transform as data_transform
import train.model as model

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

DATASET_PATH = "dataset/cfs_2017.csv"
TRACKER_PATH = os.getenv("TRACKER_PATH")

def generate_random_string(length: int) -> str:
    char = "ABCDEFGHIJKLMOPQRSTUVWXYZ1234567890"
    final = ""
    for _ in range(length):
        final += random.choice(char)
    return final

def train2():
    dataloader = data_loader.Disk(DATASET_PATH, CommodityFlow, chunk=1000)
    datacleaner = data_cleaner.DataFrame()
    datatransform = data_transform.DataTransform(CommodityFlow)

    generated_run_name = generate_random_string(6)
    experiment_name = "humamtest"
    mlflow.set_tracking_uri(uri=TRACKER_PATH)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=generated_run_name):
        psm = (PreprocessScenarioManager()
            .set_tracking(TRACKER_PATH, experiment_name)
            .set_dataloader(dataloader)
            .set_datacleaner(datacleaner)
            .set_datatransform(datatransform)
            .preprocess()
            .get_run_name())
        (ModelScenarioManager()
            .set_tracking(TRACKER_PATH, experiment_name)
            .set_dataloader(dataloader)
            .load_data(psm)
            .add_model(RandomForestRegressor())
            .add_model(GradientBoostingRegressor())
            .add_model(KNeighborsRegressor())
            .add_model(LinearRegression())
            .add_model(ElasticNet())
            .add_model(Lasso())
            .add_model(DecisionTreeRegressor())
            .train()
            .get_potential_candidates(parent_run_id, 3)
            .test_runs(parent_run_id)
            .tag_champion(parent_run_id))

def describer():
    experiment_name = "humamtest"
    generated_run_name = generate_random_string(6)
    mlflow.set_tracking_uri(uri=TRACKER_PATH)
    mlflow.set_experiment(experiment_name)
    dataloader = data_loader.Disk(DATASET_PATH, CommodityFlow, chunk=200_000)
    with mlflow.start_run(run_name=generated_run_name) as parent_run:
        mlflow.set_tag("intention", "describe")
        (PreprocessScenarioManager()
            .set_tracking(TRACKER_PATH, experiment_name)
            .set_dataloader(dataloader)
            .describe_dataset())
    return

def train():
    experiment_name = "humamtest"
    dataloader = data_loader.Disk(DATASET_PATH, CommodityFlow, chunk=200_000)
    msm = (ModelScenarioManager()
        .set_dataloader(dataloader)
        .load_data(experiment_name))

if __name__ == "__main__":
    describer()
