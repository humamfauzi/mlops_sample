import mlflow
from mlflow.models import infer_signature

import time
import random
from train.data_loader import TabularDataLoader
from train.data_cleaner import TabularDataCleaner
from train.data_transform import TabularDataTransform
from train.model import TabularModel
from typing import Optional
from train.sstruct import FeatureTargetPair, Pairs, Stage
from sklearn.metrics import mean_squared_error

# all run initiate here
# all run component should be included here and called in
# desired scenario
class PreprocessScenarioManager:
    def __init__(self):
        self.dataloader: Optional[TabularDataLoader] = None
        self.datacleaner: Optional[TabularDataCleaner] = None
        self.datatransform: Optional[TabularDataTransform] = None
        self.experiment_name: Optional[str] = None
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

    def get_run_name(self) -> str:
        return self.run_name

    def preprocess(self):
        if self.dataloader is None:
            raise ValueError("Require a data loader")
        if self.datacleaner is None:
            raise ValueError("Require a data cleaner")
        if self.datatransform is None:
            raise ValueError("Require a data cleaner")
        run_name = self.generate_name()
        with mlflow.start_run(run_name = run_name , nested=True):
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
            self.dataloader.save_pairs(run_name, pairs)
            self.run_name = run_name
        return self

class ModelScenarioManager:
    def __init__(self):
        self.dataloader: Optional[TabularDataLoader] = None
        self.pairs: Optional[Pairs] = None
        self.model_list = []
        return

    def set_dataloader(self, dataloader: TabularDataLoader):
        self.dataloader = dataloader
        return self

    def load_data(self, directory: str):
        if self.dataloader is None: 
            raise ValueError("data loader is not exist")
        self.pairs = self.dataloader.load_pairs(directory)
        return self

    def generate_name(self):
        char = "1234567890ABCDEF"
        cchar = ""
        for _ in range(4):
            cchar += random.choice(char)
        return f"MOD-{cchar}"

    def set_tracking(self, path, name):
        self.tracking_path = path
        self.experiment_name = name
        mlflow.set_tracking_uri(uri=self.tracking_path)
        mlflow.set_experiment(self.experiment_name)
        return self

    def pick_pair(self, stage: Stage) -> FeatureTargetPair:
        if self.pairs is None:
            raise ValueError("pairs not exist")
        if stage == Stage.TRAIN:
            if self.pairs.train is None:
                raise ValueError("train pairs cannot empty")
            return self.pairs.train
        if stage == Stage.VALID:
            if self.pairs.valid is None:
                raise ValueError("valid pairs cannot empty")
            return self.pairs.valid
        if stage == Stage.TEST:
            if self.pairs.valid is None:
                raise ValueError("valid pairs cannot empty")
            return self.pairs.test

    def add_model(self, model):
        self.model_list.append(model)
        return self

    def train(self):
        for mod in self.model_list:
            run_name = self.generate_name()
            with mlflow.start_run(run_name = run_name , nested=True):
                model_name = mod.__class__.__name__
                ftp = self.pick_pair(Stage.TRAIN)
                mod.fit(ftp.x_array(), ftp.y)
                mlflow.log_metric("total_trained", ftp.X.shape[0])
                mlflow.log_metric("total_features", ftp.X.shape[1])
                self.check_mse_against(mod, Stage.TRAIN)
                self.check_mse_against(mod, Stage.VALID)
                mlflow.sklearn.log_model(
                    sk_model=mod,
                    artifact_path="model",
                    signature=infer_signature(ftp.x_array(), mod.predict(ftp.x_array())),
                    input_example=ftp.x_array()[:5],
                    registered_model_name=f"{run_name}/{model_name}",
                )
                mlflow.set_tag("purpose", "model")
        return self

    def get_potential_candidates(self, num: int):
        '''
        Based on the run, choose number of candidate that came as top.
        '''
        if self.experiment_name is None:
            raise ValueError("experiment name should exist; use .set_tracking()")
        result = mlflow.search_runs(
            experiment_name = self.experiment_name,
            order_by=[f"metrics.{Stage.VALID.mse_metrics()} DESC"],
        )
        print(">>>>>>", result)
        return self

    def tag_champion(self):
        
        return self

    def check_mse_against(self, mod, stage: Stage):
        ftp = self.pick_pair(stage)
        mse = mean_squared_error(ftp.y, mod.predict(ftp.x_array()))
        mlflow.log_metric(stage.mse_metrics(), mse)
        return self

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
