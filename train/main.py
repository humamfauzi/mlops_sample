from train.scenario_manager import ScenarioManager, PreprocessScenarioManager
from train.column import CommodityFlow
import os
import random
import mlflow

import train.data_loader as data_loader
import train.data_cleaner as data_cleaner
import train.data_transform as data_transform
import train.model as model

DATASET_PATH = "dataset/cfs_2017.csv"
TRACKER_PATH = os.getenv("TRACKER_PATH")

def generate_random_string(length: int) -> str:
    char = "ABCDEFGHIJKLMOPQRSTUVWXYZ1234567890"
    final = ""
    for _ in range(length):
        final += random.choice(char)
    return final

def train2():
    dataloader = data_loader.Disk(DATASET_PATH, CommodityFlow, chunk=200_000)
    datacleaner = data_cleaner.DataFrame()
    datatransform = data_transform.DataTransform(CommodityFlow)
    generated_run_name = generate_random_string(6)
    experiment_name = "humamtest"
    mlflow.set_tracking_uri(uri=TRACKER_PATH)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=generated_run_name):
        (PreprocessScenarioManager()
            .set_dataloader(dataloader)
            .set_datacleaner(datacleaner)
            .set_datatransform(datatransform)
            .set_tracking(TRACKER_PATH, experiment_name)
            .preprocess())



def train():
    dataloader = data_loader.Disk(DATASET_PATH, CommodityFlow, chunk=200_000)
    datacleaner = data_cleaner.DataFrame()
    datatransform = data_transform.DataTransform(CommodityFlow)
    generated_run_name = generate_random_string(6)
    mmodel = model.Model().set_scenario(model.ModelScenario.MULTI_REGRESSION)
    (ScenarioManager()
        .set_run_name(RUN_NAME)
        .set_tracking(TRACKER_PATH, "humamtest")
        .start_run(generated_run_name)
        .set_dataloader(dataloader)
        .set_datacleaner(datacleaner)
        .set_datatransform(datatransform)
        .set_train(mmodel)
        .default_path()
        .end_run()
     )

if __name__ == "__main__":
    train2()
