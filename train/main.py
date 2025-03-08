from train.scenario_manager import PreprocessScenarioManager, ModelScenarioManager
from column.cfs2017 import CommodityFlow
import os
import random
import mlflow

import train.data_loader as data_loader
import train.data_cleaner as data_cleaner
import train.data_transform as data_transform
import train.model as model
from train.column import CommodityFlow

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import argparse

DATASET_PATH = "dataset/cfs_2017.csv"
TRACKER_PATH = os.getenv("TRACKER_PATH")

def generate_random_string(length: int) -> str:
    char = "ABCDEFGHIJKLMOPQRSTUVWXYZ1234567890"
    final = ""
    for _ in range(length):
        final += random.choice(char)
    return final

def multimodel_train():
    dataloader = data_loader.Disk(DATASET_PATH, CommodityFlow, chunk=1000)
    datacleaner = data_cleaner.DataFrame()
    datatransform = data_transform.DataTransform(CommodityFlow)

    generated_run_name = generate_random_string(6)
    experiment_name = "humamtest"
    mlflow.set_tracking_uri(uri=TRACKER_PATH)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=generated_run_name) as parent_run_id:
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

def base_train():
    experiment_name = "humamtest"
    dataloader = data_loader.Disk(DATASET_PATH, CommodityFlow, chunk=200_000)
    msm = (ModelScenarioManager()
        .set_dataloader(dataloader)
        .load_data(experiment_name))

def singular_train():
    experiment_name = "humamtest"
    data_cleaner_lazy_call = (data_cleaner.DataFrameLazyCall().
        filter_column([CommodityFlow.SHIPMENT_WEIGHT]).
        remove_nan_rows())
    data_transformer = (data_transform.DataTransfromLazyCall().
        add_log_transformation(CommodityFlow.SHIPMENT_VALUE).
        add_min_max_transformation(CommodityFlow.SHIPMENT_WEIGHT)

    )
    msm = (ModelScenarioManager()
        .set_dataloader(data_loader.RandomDiskLoader(DATASET_PATH, CommodityFlow, chunk=200_000))
        .set_datacleaner(data_cleaner_lazy_call)

    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different functions.")
    parser.add_argument("function_name", type=str, help="Name of the function to run (train, multimodel_train, describer)")

    args = parser.parse_args()
    function_map = {
        "base_train": base_train,
        "multimodel_train": multimodel_train,
        "describer": describer,
    }

    if args.function_name in function_map:
        function_map[args.function_name]()
    else:
        print("Invalid function name. Choose from 'train', 'multimodel_train', 'describer'.")
    
