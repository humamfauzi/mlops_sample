
from dataclasses import dataclass, field
from typing import List, Any, Dict, Union
from datetime import timedelta
from repositories.mlflow import Repository
import numpy as np
import pickle
import time
import os
import json
from enum import Enum

class MetadataKey(Enum):
    MODEL_NAME = "model_name"
    TRAIN_ACCURACY = "train_accuracy"
    TEST_ACCURACY = "test_accuracy"
    METRICS = "metrics"
    TRAINED_ROWS = "trained_rows"
    TRAIN_TIME = "train_time"
    ALGORITHM = "algorithm"

@dataclass
class Metadata:
    """Metadata item for the model metadata response"""
    key: MetadataKey    
    display: str
    value: Any
    
    def to_dict(self) -> Dict[MetadataKey, Any]:
        return { "key": self.key, "display": self.display, "value": self.value }


@dataclass
class InputBase:
    """Base class for input items"""
    key: str
    display: str

@dataclass
class CategoricalInput(InputBase):
    """Categorical input item for the model metadata response"""
    type: str = "categorical"
    enumeration: List[str] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "key": self.key,
            "display": self.display,
            "enumeration": self.enumeration
        }


@dataclass
class NumericalInput(InputBase):
    """Numerical input item for the model metadata response"""
    type: str = "numerical"
    min: float = None
    max: float = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type,
            "key": self.key,
            "display": self.display
        }
        if self.min is not None:
            result["min"] = self.min
        if self.max is not None:
            result["max"] = self.max
        return result

Input = Union[CategoricalInput, NumericalInput]

@dataclass
class NumericalOutput:
    value: float
    time_elapsed: timedelta

class CategoricalOutput:
    value: str
    probability: float
    time_elapsed: timedelta

Output = Union[NumericalOutput, CategoricalOutput]


@dataclass
class ShortDescription:
    name: str
    display: str

@dataclass
class TabularModel:
    name: str
    display: str
    desc: str
    metadata: List[Metadata]
    input: List[Input]

class Model:
    information: TabularModel
    run_id: str
    preprocess_map: Dict[str, Any]
    predict_func: callable

    def __init__(self, run_id: str):
        self.run_id = run_id

    def load_metadata(self, directory: str) -> "Model":
        """
        Load metadata from json to dataclass TabularModel
        """
        metadata_dir = f"{directory}/metadata"
        metadata_files = os.listdir(metadata_dir) 
        container = {}
        for file in metadata_files:
            if file == "metadata.json":
                with open(f"{metadata_dir}/{file}", 'r') as f:
                    container = json.load(f)
        self.information = TabularModel(
            name=container["name"],
            display=container["display"],
            desc=container["desc"],
            metadata= generate_metadata_from_json(container["metadata"]),
            input= generate_input_from_json(container["input"]),
        )
        return self

    def load_model(self, directory: str) -> "Model":
        """
        Load model pickle
        """
        model_dir = f"{directory}/model"
        model_files = os.listdir(model_dir) 
        for file in model_files:
            if file == "model.pkl":
                with open(f"{model_files}/{file}", 'rb') as f:
                    self.predict_func = pickle.load(f).predict
        return self

    def load_preprocess(self, directory: str) -> "Model":
        """
        Load preprocess from directory. The preprocess is divided by key/columns in dataset.
        Each column might have multiple preprocess therefore it is likely to have more that one pickle 
        in each column. We try our best to keep all process commutative.
        """
        self.preprocess_map = {}
        preprocess_directory = f"{directory}/preprocess"
        preprocess_key = os.listdir(preprocess_directory)
        for key in preprocess_key:
            available_preprocess = os.listdir(f"{preprocess_directory}/{key}")
            for preprocess in available_preprocess:
                function_container = [
                    lambda x: x,
                ]
                with open(f"{preprocess_directory}/{key}/{preprocess}", 'rb') as f:
                    function_container.append(pickle.load(f).transform)
                self.preprocess[key] = function_container
        return self

    def load_postprocess(self, directory: str) -> "Model":
        """
        Similar to preprocess but this is only applies to a column. A tabular machine learning only
        have one output therefore there is no need to iterate for column
        """
        self.postprocess = [
            lambda x: x,
        ]
        postprocess_dir = f"{directory}/postprocess"
        preprocess_files = os.listdir(postprocess_dir) 
        for file in preprocess_files:
            with open(f"{postprocess_dir}/{file}", 'rb') as f:
                self.postprocess.append(pickle.load(f).transform)
        return self

    def preprocess(self, data: Dict[str,Any]):
        """
        Iterating key and value in data and apply preprocess if the key is in preprocess dictionary.
        """
        result = {}
        for key, value in data.items():
            if key in self.preprocess.keys():
                for preprocess in self.preprocess[key]:
                    value = preprocess(value)
            result[key] = value
        return result

    def infer(self, data: Dict[str, Any]) -> NumericalOutput:
        """
        Infer would 
        1. preprocess data using preprocess function that loaded via load_preprocess method
        2. predict the data using predict function that loaded via load_model method
        3. postprocess the result using postprocess function that loaded via load_postprocess method
        4. return the result and inference time
        """
        input = self.preprocess(data)
        start_time = time.time()
        result = self.predict(input)
        result = self.postprocess(result)
        elapsed = time.time() - start_time
        result.time_elapsed = timedelta(seconds=elapsed)
        return NumericalOutput(value=result, time_elapsed=elapsed) 

    def short_description(self) -> ShortDescription:
        return ShortDescription(
            name=self.information.name,
            display=self.information.display,
        )
    def validate_input(self, data: Dict[str, Any]):
        container = {}
        for i in self.information.input:
            if i not in data.keys():
                return [{}, f"Key {i} is missing"]

        for key, value in data.items():
            for i in self.information.input:
                if i.key == key:
                    if isinstance(i, CategoricalInput):
                        if value not in i.enumeration:
                            return [{}, f"Value {value} is not in enumeration for key {key}"]
                        container[i] = float(value)
                    elif isinstance(i, NumericalInput):
                        if i.min is not None and value < i.min:
                            return [{}, f"Value {value} is less than minimum for key {key}"]
                        if i.max is not None and value > i.max:
                            return [{}, f"Value {value} is greater than maximum for key {key}"]
                        container[i] = value
        return [container, ""]

class Sample:
    def __init__(self): pass
class ModelRepository:
    def __init__(self, repository: Repository):
        self.models = []
        self.repository = repository

    def get(self) -> List[str]:
        pass

    def metadata(self, model_name: str) -> TabularModel:
        model = self._find_model(model_name)
        return model.information

    def load(self) -> "ModelRepository":
        runs = self.client.search_runs( experiment_ids=[self.experiment_id], filter_string="tags.production = 'true'",)
        experiment_id = mlflow.get_experiment_by_name(self.experiment).experiment_id
        for run in runs:
            uri = f"mlflow-artifacts:/{experiment_id}/{run.info.run_id}/artifacts"
            destination_path = "artifacts/{run.info.run_id}"
            mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=destination_path)
            model = (Model(run.info.run_id)
                .load_preprocess(destination_path)
                .load_model(destination_path)
                .load_metadata(destination_path))
            self.models.append(model)
        return self

    def infer(self, model_name: str, data: Dict[str, Any]) -> Output:
        model = self._find_model(model_name)
        return model.predict(data)

    def list(self) -> List[ShortDescription]:
        return [model.short_description() for model in self.models]

    def _find_model(self, model_name: str) -> "Model":
        for i in model_name:
            if i.information.name == model_name:
                return i
        return None

    def validate_input(self, model_name: str, data: Dict[str, Any]) -> List[Union[Dict[str, Any], str]]:
        model = self._find_model(model_name)
        return model.validate_input(data)


def generate_metadata_from_json(metadata: Dict[str, Any]) -> List[Metadata]:
    result = []
    for key, value in metadata.items():
        result.append(Metadata(key=key, display=value["display"], value=value["value"]))
    return result

def generate_input_from_json(input: Dict[str, Any]) -> List[Input]:
    result = []
    for key, value in input.items():
        if value["type"] == "categorical":
            result.append(CategoricalInput(
                key=key,
                display=value["display"],
                enumeration=value["enumeration"]
            ))
        elif value["type"] == "numerical":
            result.append(NumericalInput(
                key=key,
                display=value["display"],
                min=value.get("min", None),
                max=value.get("max", None)
            ))
    return result


class ModelServer:
    def __init__(self, repository: Repository):
        self.repository = repository

    def load(self) -> "ModelServer":
        parent_runs = self.repository.get_all_available_runs()
        for run in parent_runs:
            self.repository.load_run(run)
        return self

    def list() -> List[str]: return []

    def metadata(self, model_name: str) -> TabularModel: pass

    def infer(self, model_name: str, data: Dict[str, Any]) -> Output: pass