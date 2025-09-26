import os
from abc import ABC, abstractmethod
from enum import Enum

import time
import random
import pickle
from dataclasses import dataclass
from typing import List, Optional
from train.data_io import Disk
from train.data_cleaner import Cleaner
from train.data_transform import Transformer
from train.dataset import TrackingDataset
from train.model import TabularModel
from typing import Optional
from train.sstruct import FeatureTargetPair, Pairs, Stage
from repositories.dummy import DummyMLflowRepository as Repository, Manifest
from sklearn.metrics import mean_squared_error

# all run initiate here
# all run component should be included here and called in
# desired scenario
class PreprocessScenarioManager:
    def __init__(self):
        self.dataloader: Optional[Disk] = None
        self.datacleaner: Optional[Cleaner] = None
        self.datatransform: Optional[TabularDataTransform] = None
        self.repository: Optional[Repository] = None
        self.experiment_name: Optional[str] = None
        return
    
    # GETTER & SETTER
    def set_dataloader(self, dl: Disk):
        self.dataloader = dl
        return self

    def set_repository(self, repository: Repository):
        self.repository = repository
        return self

    def set_datacleaner(self, dc: Cleaner):
        self.datacleaner = dc
        return self

    def set_datatransform(self, dt: Transformer):
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
        self.repository: Optional[Repository] = None
        self.pairs: Optional[Pairs] = None
        self.model_list = []
        return

    def set_repository(self, repository: Repository):
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

    def add_model(self, model, desc: str):
        self.model_list.append((model, desc))
        return self

    def train(self):
        for mod, desc in self.model_list:
            metadata_manifest = []
            run_name = self.generate_name()
            self.repository.start_nested_run(run_name)
            model_name = mod.__class__.__name__
            ftp = self.pick_pair(Stage.TRAIN)
            start = time.time()
            mod.fit(ftp.x_array(), ftp.y_array())

            (self.repository
            .log_metric("total_trained", ftp.X.shape[0])
            .log_metric("total_trained", ftp.X.shape[0])
            .log_metric("total_features", ftp.X.shape[1])
            .set_tag("purpose", "model")
            .set_tag("level", "candidate")
            .set_tag("description", desc)
            .set_tag("algorithm", model_name))

            metadata_manifest.extend([
                Manifest.create_metadata_item("trained_rows", ftp.X.shape[0]).to_dict(),
                Manifest.create_metadata_item("total_features", ftp.X.shape[1]).to_dict(),
                Manifest.create_metadata_item("description", desc).to_dict(),
                Manifest.create_metadata_item("algorithm", model_name).to_dict(),
                Manifest.create_metadata_item("train_duration", time.time() - start).to_dict(),
                Manifest.create_metadata_item("model_name", run_name).to_dict(),
                Manifest.create_metadata_item("ordering", [c for c in ftp.X.columns]).to_dict(),
            ])

            (self.check_mse_against(mod, Stage.TRAIN, metadata_manifest)
            .check_mse_against(mod, Stage.VALID, metadata_manifest))

            self.repository.save_model(mod, self.repository.get_parent_run_id(), run_name)
            self.repository.save_model_manifest(metadata_manifest, self.repository.get_parent_run_id(), run_name)
            self.repository.end_nested_run()
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
        self.repository.set_tag_run(first["run_id"], "level", "test")
        return self

    def test_runs(self):
        child_runs = self.repository.find_child_runs(filt={"level": "test"}, order_by="created")
        if len(child_runs) == 0:
            raise ValueError("No child runs found for test runs")
        first = child_runs.iloc[0] 
        mod = self.repository.load_model(self.repository.get_parent_run_id(), first["tags.mlflow.runName"])
        return self.check_mse_against(mod, Stage.TEST, [])

    def tag_champion(self):
        child_runs = self.repository.find_child_runs(
            order_by=f"metrics.{Stage.mse_metrics(Stage.TEST)}",
            filt={"level": "test"}
        )
        best = child_runs.iloc[0]
        self.repository.set_tag_run(best["run_id"], "level", "champion")
        self.repository.set_tag_run(self.repository.get_parent_run_id(), "stage", "production")
        return self

    def check_mse_against(self, mod, stage: Stage, metadata_manifest: Optional[Manifest] = []):
        ftp = self.pick_pair(stage)
        mse = mean_squared_error(ftp.y_array(), mod.predict(ftp.x_array()))
        self.repository.log_metric(Stage.mse_metrics(stage), mse)
        metadata_manifest.append( Manifest.create_metadata_item(Stage.mse_metrics(stage), mse).to_dict())
        return self



class InstructionFactory:
    def __init__(self):
        pass

    @classmethod
    def parse_instruction(cls, instruction: dict):
        instructions = []
        for inst in instruction["instructions"]:
            instructions.append(InstructionStep(
                type=InstructionEnum.from_string(inst["type"]),
                properties=inst.get("properties", {}),
                call=inst.get("call", [])
            ))
        return Instruction(
            name=instruction["name"],
            description=instruction["description"],
            instructions=instructions
        )

class InstructionEnum(Enum):
    DATA_IO = "data_io"
    DATA_CLEANER = "data_cleaner"
    DATA_TRANSFORMER = "data_transformer"
    MODEL_TRAINER = "model_trainer"

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Unknown InstructionEnum value: {value}")

@dataclass
class InstructionStep:
    type: InstructionEnum
    properties: dict
    # let the respective class handle the parsing of dictionary
    call: List[dict]

@dataclass
class Instruction:
    name: str
    description:str
    instructions: List[InstructionStep]

class ScenarioManager:
    def __init__(self, instruction: Instruction):
        self.instruction = instruction
        self.pipeline: List[any] = []
    def construct(self):
        for step in self.instruction.instructions:
            if step.type == InstructionEnum.DATA_IO:
                self.pipeline.append(Disk.parse_instruction(step.properties, step.call))
            elif step.type == InstructionEnum.DATA_CLEANER:
                self.pipeline.append(Cleaner.parse_instruction(step.properties, step.call))
            elif step.type == InstructionEnum.DATA_TRANSFORMER:
                self.pipeline.append(Transformer.parse_instruction(step.properties, step.call))
            elif step.type == InstructionEnum.MODEL_TRAINER:
                self.pipeline.append(Trainer.parse_instruction(step.properties))
        return self
    def execute(self):
        recurse = None
        for component in self.pipeline:
            recurse = component.execute(recurse)
        return recurse
    