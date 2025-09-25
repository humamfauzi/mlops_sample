from .scenario_manager import PreprocessScenarioManager, ModelScenarioManager
import os
from . import data_io, data_cleaner, data_transform
from column.cfs2017 import CommodityFlow
import argparse

DATASET_PATH = "dataset"
TRACKER_PATH = os.getenv("TRACKER_PATH")


def multimodel_train():
    pass

def describer():
    pass

def base_train():
    experiment_name = "humamtest"
    dataloader = data_io.Disk(DATASET_PATH, CommodityFlow, chunk=200_000)
    msm = (ModelScenarioManager()
        .set_dataloader(dataloader)
        .load_data(experiment_name))

def singular_train():
    '''
    This is the base train function. It is used to train a single model.
    It only use shipment weight as a feature. Use this function to verify the training process.
    '''
    experiment_name = "humamtest_lazycall"

    repository = Repository(TRACKER_PATH, experiment_name)
    data_loader_preprocess = (data_io.Disk(DATASET_PATH, "cfs_2017", repository=repository)
        .load_dataframe_via_csv(CommodityFlow, {"nrows": 200_000})
        .save_pair_via_parquet())
    data_loader_train = (data_io.Disk(DATASET_PATH, "cfs_2017", repository=repository)
        .load_pair_via_parquet())
    data_cleaner_lc = (data_cleaner.DataFrameLazyCall().
        filter_columns([CommodityFlow.SHIPMENT_WEIGHT, CommodityFlow.SHIPMENT_VALUE]).
        remove_nan_rows())
    data_transformer_lc = (data_transform.DataTransformLazyCall(CommodityFlow, repository=repository).
        add_log_transformation(CommodityFlow.SHIPMENT_VALUE, data_transform.TransformationMethods.REPLACE).
        add_min_max_transformation(CommodityFlow.SHIPMENT_WEIGHT, data_transform.TransformationMethods.REPLACE))
    repository.start()
    (PreprocessScenarioManager()
        .set_repository(repository)
        .set_dataloader(data_loader_preprocess)
        .set_datacleaner(data_cleaner_lc)
        .set_datatransform(data_transformer_lc)
        .preprocess()
        .get_run_name())
    (ModelScenarioManager()
        .set_repository(repository)
        .set_dataloader(data_loader_train)
        .load_data()
        .add_model(LinearRegression(), "A basic Linear Regression. Using to test the training sequence from training to deployment")
        .train()
        .get_potential_candidates()
        .test_runs()
        .tag_champion())
    repository.set_tag("base", "singular_train")
    repository.stop()
    return

def list_all_possible_instructions():
    folder_path = "../train_config"
    files = os.listdir(folder_path)
    print("Available instructions:", files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different functions.")
    parser.add_argument("instruction", type=str, help="Name of the instruction set.")
    parser.add_argument("instruction_list", help="list all available instructions", action="store_true")
    args = parser.parse_args()

    if args.instruction_list:
        list_all_possible_instructions()
    elif args.instruction:
        call_instruction(args.instruction)

