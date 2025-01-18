import mlflow
import time
import random
from train.data_loader import TabularDataLoader
from train.data_cleaner import TabularDataCleaner
from train.data_transform import TabularDataTransform
from train.model import TabularModel
from typing import Optional
# all run initiate here
# all run component should be included here and called in
# desired scenario
class PreprocessScenarioManager:
    def __init__(self):
        self.dataloader: Optional[TabularDataLoader] = None
        self.datacleaner: Optional[TabularDataCleaner] = None
        self.datatransform: Optional[TabularDataTransform] = None
        return

    def set_dataloader(self, dl: TabularDataLoader):
        self.dataloader = dl
        return self
    def set_datacleaner(self, dc: TabularDataCleaner):
        self.datacleaner = dc
        return self
    def set_datatransform(self, dt: TabularDataTransform):
        self.datatransform = dt
        return self

    def set_tracking(self, path, name):
        self.tracking_path = path
        self.experiment_name = name
        mlflow.set_tracking_uri(uri=self.tracking_path)
        mlflow.set_experiment(self.experiment_name)
        return self

    def generate_name(self):
        char = "1234567890ABCDEF"
        cchar = ""
        for _ in range(4):
            cchar += random.choice(char)
        return f"PRE-{cchar}"

    def preprocess(self):
        if self.dataloader is None:
            raise ValueError("Require a data loader")
        if self.datacleaner is None:
            raise ValueError("Require a data cleaner")
        if self.datatransform is None:
            raise ValueError("Require a data cleaner")
        run_name = self.generate_name()
        with mlflow.start_run(run_name = run_name , nested=True) as child_run:
            start = time.time()
            df = self.dataloader.load_data()
            mlflow.log_param("origin_size", df.shape)
            df = self.datacleaner.clean_data(df)
            mlflow.log_param("clean_size", df.shape)
            pairs = self.datatransform.transform_data(df)
            mlflow.log_params({
                "train_feature_size": pairs.train.X.shape, "train_target_size": pairs.train.y.shape,
                "valid_feature_size": pairs.valid.X.shape, "valid_target_size": pairs.valid.y.shape,
                "test_feature_size": pairs.test.X.shape, "test_target_size": pairs.test.y.shape,
            })
            mlflow.set_tag("purpose", "preprocess")
            mlflow.log_metric("duration", time.time() - start)
            self.dataloader.save_data(pairs.train.X, f"{run_name}/train/feature.parquet")
            self.dataloader.save_data(pairs.train.y.to_frame(), f"{run_name}/train/target.parquet")
            self.dataloader.save_data(pairs.valid.X, f"{run_name}/valid/feature.parquet")
            self.dataloader.save_data(pairs.valid.y.to_frame(), f"{run_name}/valid/target.parquet")
            self.dataloader.save_data(pairs.test.X, f"{run_name}/test/feature.parquet")
            self.dataloader.save_data(pairs.test.y.to_frame(), f"{run_name}/test/target.parquet")
            print("child run id", child_run.info.run_id)
        return self

class ModelScenarioManager:
    def __init__(self):
        self.dataloader: Optional[TabularDataLoader] = None
        pass

# deprecated; should either use PreprocessScenarioManager or ModelScenarioManager
class ScenarioManager:
    def __init__(self):
        self.dataloader: Optional[TabularDataLoader] = None
        self.datacleaner: Optional[TabularDataCleaner] = None
        self.datatransform: Optional[TabularDataTransform] = None
        self.model: Optional[TabularModel] = None
        self.run_name: Optional[str] = None
        return

    def set_run_name(self, name: str):
        self.run_name = name
        return self

    def set_dataloader(self, dataloader: TabularDataLoader):
        self.dataloader = dataloader
        return self

    def set_datacleaner(self, datacleaner: TabularDataCleaner):
        self.datacleaner = datacleaner
        return self

    def set_datatransform(self, datatransform: TabularDataTransform):
        if self.run_name is None:
            raise ValueError("run name is required for data transform tracking")
        self.datatransform = datatransform
        self.datatransform.set_run_name(self.run_name)
        return self

    def set_train(self, train: TabularModel):
        self.model = train
        self.model.set_run_name(self.run_name)
        self.model.set_tracker(self.is_using_tracker)
        return self

    def set_tracking(self, path, name):
        self.tracking_path = path
        self.experiment_name = name
        mlflow.set_tracking_uri(uri=self.tracking_path)
        mlflow.set_experiment(self.experiment_name)
        return self

    def start_run(self, run_name):
        mlflow.start_run(run_name=run_name)
        self.is_using_tracker = True
        return self

    def end_run(self):
        mlflow.end_run()

    def default_path(self):
        df = self.dataloader.load_data()
        df_cleaned = self.datacleaner.clean_data(df)
        pairs = self.datatransform.transform_data(df_cleaned)
        self.model.train_data(pairs)
        return self
