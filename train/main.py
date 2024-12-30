from train.scenario_manager import ScenarioManager
from train.column import CommodityFlow

import train.data_loader as data_loader
import train.data_cleaner as data_cleaner
import train.data_transform as data_transform
import train.model as model

DATASET_PATH = "dataset/cfs_2017.csv"
TRACKER_PATH = "http://mlflow:5000" # see docker compose for details
RUN_NAME = "base_run"

def train():
    dataloader = data_loader.Disk(DATASET_PATH, CommodityFlow)
    datacleaner = data_cleaner.DataFrame()
    datatransform = data_transform.DataTransform(CommodityFlow)
    mmodel = (model.Model()
        .set_train_name("humamtest"))
    (ScenarioManager()
        .set_run_name(RUN_NAME)
        .set_tracking(TRACKER_PATH, "humamtest")
        .start_run("long_run")
        .set_dataloader(dataloader)
        .set_datacleaner(datacleaner)
        .set_datatransform(datatransform)
        .set_train(mmodel)
        .default_path()
        .end_run()
     )

if __name__ == "__main__":
    train()
