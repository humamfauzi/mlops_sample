import mlflow
from mlflow.models import infer_signature
import os

import time
import random
import pickle
from train.data_io import Disk
from train.data_cleaner import TabularDataCleaner
from train.data_transform import TabularDataTransform
from train.dataset import TrackingDataset
from train.model import TabularModel
from typing import Optional
from train.sstruct import FeatureTargetPair, Pairs, Stage
from repositories.mlflow import MLflowRepository
from sklearn.metrics import mean_squared_error

# all run initiate here
# all run component should be included here and called in
# desired scenario
class PreprocessScenarioManager:
    def __init__(self):
        self.dataloader: Optional[Disk] = None
        self.datacleaner: Optional[TabularDataCleaner] = None
        self.datatransform: Optional[TabularDataTransform] = None
        self.repository: Optional[MLflowRepository] = None
        self.experiment_name: Optional[str] = None
        return
    
    # GETTER & SETTER
    def set_dataloader(self, dl: Disk):
        self.dataloader = dl
        return self

    def set_repository(self, repository: MLflowRepository):
        self.repository = repository
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

    ## primara function
    def describe_dataset(self):
        if self.dataloader is None:
            raise ValueError("Require a data loader")
        start = time.time()
        describe = self.dataloader.describe_all_numerical_data()
        dataset = TrackingDataset("/dataset/cfs2017", "cfs2017", "Commodity Flow Survey 2017")
        with mlflow.start_run(run_name = "ASD" , nested=True):
            mlflow.set_tag("purpose", "describe")
            mlflow.log_metric("duration", time.time() - start)
            for col in describe:
                dataset.set_numerical_column_properties(
                    col, 
                    describe[col]["mean"], 
                    describe[col]["count"],
                    describe[col]["stddev"], 
                    describe[col]["sum"],
                    describe[col]["min"],
                    describe[col]["max"],
                )
            mlflow.log_input(dataset, context="description")
        return self

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
            self.dataloader.save_data(pairs)
            self.run_name = run_name
        return self

    def _save_manifest(self, path):
        pass

class ModelScenarioManager:
    def __init__(self):
        self.dataloader: Optional[Disk] = None
        self.repository: Optional[MLflowRepository] = None
        self.pairs: Optional[Pairs] = None
        self.model_list = []
        return

    def set_repository(self, repository: MLflowRepository):
        self.repository = repository
        return self

    def set_dataloader(self, dataloader: Disk):
        self.dataloader = dataloader
        return self

    def load_data(self):
        if self.dataloader is None: 
            raise ValueError("data loader is not exist")
        self.pairs = self.dataloader.load_data()
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
                mod.fit(ftp.x_array(), ftp.y_array())
                mlflow.log_metric("total_trained", ftp.X.shape[0])
                mlflow.log_metric("total_features", ftp.X.shape[1])
                mlflow.set_tag("purpose", "model")
                mlflow.set_tag("level", "candidate")
                self.check_mse_against(mod, Stage.TRAIN)
                self.check_mse_against(mod, Stage.VALID)
                self._save_processing_as_blob("artifacts/models", model_name, mod)
        self._save_models()
        return self

    def _save_models(self):
        run = mlflow.active_run()
        dirr = "artifacts/models"
        client = mlflow.tracking.MlflowClient()
        child_run = client.get_run(run.info.run_id)
        parent_run_id = child_run.data.tags.get("mlflow.parentRunId")
        mlflow.log_artifacts(run_id=parent_run_id, local_dir=dirr, artifact_path="models")
        return self

    def _save_processing_as_blob(self, directory, model_name, model_object):
        os.makedirs(f'{directory}' , exist_ok=True)
        path = f'{directory}/{model_name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(model_object, f)
        return path

    def get_potential_candidates(self):
        '''
        Based on the run, choose number of candidate that came as top.
        '''
        child_runs = self.repository.find_child_runs(filt={"purpose": "model"}, order_by=f"metrics.{Stage.mse_metrics(Stage.VALID)}")
        if len(child_runs) == 0:
            raise ValueError("No child runs found for potential candidates")
        first = child_runs.iloc[0] 
        self.repository.set_tag(first["run_id"], "level", "test")
        return self

    def compose_filter_string(self, ddict):
        final = []
        for key, value in ddict.items():
            final.append(f"tags.{key} = '{value}'")
        return " AND ".join(final)

    def test_runs(self):
        child_runs = self.repository.find_child_runs(filt={"level": "test"}, order_by="created")
        if len(child_runs) == 0:
            raise ValueError("No child runs found for test runs")
        first = child_runs.iloc[0] 
        mod = self.load_model(first["tags.mlflow.runName"])
        self.check_mse_against(mod, Stage.TEST)
        return self

    # TODO : Create a manifest in the parent run so it can read the manifest
    # and load the model. Manifest should contain all the model name and the
    # path to the model
    def load_model(self, child_run_name: str):
        model_path = self.repository.compose_model_path(child_run_name=child_run_name)
        return self.repository.load_model(model_path)
    
    def _create_manifest(self, parent_run_id: str):
        path = "artifacts/model_manifest.json"
        client = mlflow.tracking.MlflowClient()
        client.log_artifact(run_id=parent_run_id, local_path=path)

    def tag_champion(self):
        child_runs = self.repository.find_child_runs(
            order_by=f"metrics.{Stage.mse_metrics(Stage.TEST)}",
            filt={"level": "test"}
        )
        best = child_runs[0]
        self.repository.set_tag(best.info.run_id, "level", "champion")
        return self

    def check_mse_against(self, mod, stage: Stage):
        ftp = self.pick_pair(stage)
        mse = mean_squared_error(ftp.y_array(), mod.predict(ftp.x_array()))
        mlflow.log_metric(Stage.mse_metrics(stage), mse)
        return self

# deprecated; should either use PreprocessScenarioManager or ModelScenarioManager
class ScenarioManager:
    def __init__(self):
        self.dataloader: Optional[Disk] = None
        self.datacleaner: Optional[TabularDataCleaner] = None
        self.datatransform: Optional[TabularDataTransform] = None
        self.model: Optional[TabularModel] = None
        self.run_name: Optional[str] = None
        return

    def set_run_name(self, name: str):
        self.run_name = name
        return self

    def set_dataloader(self, dataloader:Disk):
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
