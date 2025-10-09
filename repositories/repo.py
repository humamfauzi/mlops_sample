import math
from random import random
from xml.parsers.expat import model
from repositories.sqlite import SQLiteRepository
from repositories.disk import Disk
import repositories.noop as noop
from dataclasses import dataclass
from typing import List

class Facade:
    @classmethod
    def parse_instruction(cls, config: dict):
        if len(config) == 0:
            print("No configuration provided, using default noop repositories.")
            return cls("sample", noop.Repository(), noop.Object())
        repository = config.get("data", None)
        if repository.get("type") == "sqlite":
            prop = repository.get("properties", {})
            r = SQLiteRepository(**prop)
        else:
            raise ValueError(f"Unknown repository type: {repository.get('type')}")
        objectt = config.get("object", None)
        if objectt.get("type") == "disk":
            prop = objectt.get("properties", {})
            o = Disk(**prop)
        elif objectt.get("type") == "s3":
            from repositories.s3 import S3
            prop = objectt.get("properties", {})
            o = S3(**prop)
        else:
            raise ValueError(f"Unknown object store type: {objectt.get('type')}")
        experiment_id = config.get("experiment_id", "sample")
        return cls(experiment_id, r, o)

    def __init__(self, experiment_id: str, repository=noop.Repository(), object_store=noop.Object()):
        self.experiment_id = experiment_id
        self.repository = repository
        self.object_store = object_store

    def new_experiment(self, name: str):
        self.repository.new_experiment(name)
        return self

    def list_all_experiments(self):
        return self.repository.list_all_experiments()

    def new_run(self, name: str):
        new_run_id = self.repository.new_run(name, self.experiment_id)
        self.current_run_id = new_run_id
        return self

    def set_intent(self, intent: str):
        self.repository.new_property(self.current_run_id, "name.intent", intent)
        return self

    def set_description(self, description: str):
        self.repository.new_property(self.current_run_id, "name.description", description)
        return self

    def generate_run_id(self):
        source = "ABCDEFGHIJKLMNOPQRSTUVWXYZ12345678890"
        return ''.join([source[math.floor(random()*len(source))] for _ in range(6)])

    def new_child_run(self, name: str):
        new_run_id = self.repository.new_child_run(name, self.current_run_id, self.experiment_id)
        self.current_child_run_id = new_run_id
        return self

    def set_row_size(self, size: int):
        self.repository.new_property(self.current_run_id, "size.load.row", str(size))
        return self

    def set_column_size(self, size: int):
        self.repository.new_property(self.current_run_id, "size.load.column", str(size))
        return self

    def set_dataset_name(self, name: str):
        self.repository.new_property(self.current_run_id, "name.dataset", name)
        return self

    def set_model_filepath(self, path: str):
        self.repository.new_property(self.current_child_run_id, "name.path.model", path)
        return self

    def set_transformation_filepath(self, path: str):
        self.repository.new_property(self.current_run_id, "name.path.transformation", path)
        return self

    def set_data_loading_time(self, time: int):
        self.repository.new_metric(self.current_run_id, "time_ms.loading", time)
        return self

    def set_data_cleaning_time(self, time: int):
        self.repository.new_metric(self.current_run_id, "time_ms.cleaning", time)
        return self

    def set_row_size_after_cleaning(self, size: int):
        self.repository.new_property(self.current_run_id, "size.clean.row", str(size))
        return self

    def set_column_size_after_cleaning(self, size: int):
        self.repository.new_property(self.current_run_id, "size.clean.column", str(size))
        return self

    def set_pair_size(self, stage: str, nrows: int, ncols: int):
        self.repository.new_property(self.current_run_id, f"size.split.{stage}.row", str(nrows))
        self.repository.new_property(self.current_run_id, f"size.split.{stage}.column", str(ncols))
        return self

    def set_total_runtime(self, time: int):
        self.repository.new_metric(self.current_run_id, "time_ms.all", time)
        return self

    def set_total_transformation_time(self, time: int):
        self.repository.new_metric(self.current_run_id, "time_ms.transforming.all", time)
        return self
    
    def set_transformation_time(self, name: str, time: int):
        self.repository.new_metric(self.current_run_id, f"time_ms.transforming.{name}", time)
        return self

    def set_object_transformation(self, type: str, obj: str):
        self.repository.new_object(self.current_run_id, type, obj)
        return self

    def save_transformation_instruction(self, instructions: list):
        self.object_store.save_transformation_instruction(self.current_run_id, instructions)
        return self

    def save_transformation_object(self, transformation_objects: list):
        self.object_store.save_transformation_object(self.current_run_id, transformation_objects)
        return self
    
    def load_transformation_instruction(self, run_id: int):
        return self.object_store.load_transformation_instruction(run_id)
    
    def load_transformation_object(self, run_id: int):
        return self.object_store.load_transformation_object(run_id)
    
    def save_model(self, model):
        self.object_store.save_model(self.current_run_id, model)
        return self
    
    def load_model(self, name: str):
        return self.object_store.load_model(name)

    def set_metric(self, stage, metric_name: str, value: float):
        self.repository.new_metric(self.current_child_run_id, f"validation.{stage}.{metric_name}", value)
        return self

    def set_validation_time(self, stage: str, time: float):
        self.repository.new_metric(self.current_child_run_id, f"time_ms.validation.{stage}", time)
        return self

    def set_training_time(self, time: float):
        self.repository.new_metric(self.current_child_run_id, f"time_ms.train", time)
        return self

    def set_training_type(self, name: str):
        self.repository.new_property(self.current_child_run_id, "name.model", name)
        return self

    def set_model_properties(self, options: dict):
        for k, v in options.items():
            self.repository.new_property(self.current_child_run_id, f"property.{k}", str(v))
        return self

    def set_model_hyperparameters(self, options: dict):
        for k, v in options.items():
            self.repository.new_property(self.current_child_run_id, f"parameter.{k}", str(v))
        return self

    def find_best_model(self, metric: str):
        return self.repository.find_best_model_within_run(self.current_run_id, f"validation.valid.{metric}")

    def tag_as_the_best(self):
        self.repository.upsert_tag(self.current_child_run_id, "level", "best")
        return self

    def find_all_available_runs(self):
        return self.repository.find_all_available_runs(self.experiment_id)

    def load_model_under_parent_run(self, parent_run_id: str):
        best = self.repository.find_best_model_within_run(parent_run_id, "validation.valid.accuracy")
        if best is None:
            raise ValueError(f"No best model found under parent run {parent_run_id}")
        model_path = self.repository.find_property(best, "name.path.model")
        if model_path is None:
            raise ValueError(f"No model path found for best model {best} under parent run {parent_run_id}")
        return self.object_store.load_model(model_path)

    def get_all_published_candidates(self, experiment_id: str):
        return self.repository.get_all_published_candidates(experiment_id)

    def load_inference(self, model_name: str):
        return self.object_store.load_model(model_name)

    def get_intent(self):
        return self.repository.get_intent(self.current_run_id)

    def get_model_run_id(self, model_id: str):
        return self.repository.get_model_run_id(model_id)

    def nominate_for_publishing(self, intent: str, primary_metric: str, current_score: float, current_model_id: int):
        id, previous_score = self.repository.select_previously_published(self.experiment_id, intent, primary_metric)
        key = "status.deployment"
        if id is None:
            self.repository.upsert_tag(current_model_id,key, "published")
            return
        if current_score < previous_score:
            self.repository.upsert_tag(id,key, "retracted")
            self.repository.upsert_tag(current_model_id,key, "published")
        else:
            self.repository.upsert_tag(current_model_id,key, "inferior")

    def get_available_input(self, run_id: int):
        instruction = self.repository.get_object(run_id, "transformation.allowed_columns")
        return instruction

    def get_run_description(self, run_id: int):
        return self.repository.find_property(run_id, "name.description")

    def get_model_best_model(self, run_id: int):
        _, name = self.repository.find_tagged_best_model(run_id)
        model =  self.object_store.load_model(run_id, name)
        return model

@dataclass
class InferenceModelInstruction:
    id: str # it is a 6 random string of all caps

@dataclass
class Transformation:
    id: str

@dataclass
class InferenceTransformationInstruction:
    # available column used for parsing from query string to
    # data transformation input
    available_column: List[str]
    steps: List[Transformation]

@dataclass
class InferenceInstruction:
    model: InferenceModelInstruction
    transformation: InferenceTransformationInstruction