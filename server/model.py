
from dataclasses import dataclass, field
from typing import List, Any, Dict, Union
from datetime import timedelta
from repositories.mlflow import Repository, Manifest, InputItem, MetadataItem
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
    value: Union[float, np.ndarray]
    time_elapsed: timedelta

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.value, np.ndarray):
            self.value = [ round(i, 2) for i in self.value.tolist()]
        elif isinstance(self.value, (float, int)):
            self.value = round(self.value, 2)
        return {
            "value": self.value,
            "time_elapsed": f"{self.time_elapsed.total_seconds():.2f} seconds"
        }

class CategoricalOutput:
    value: str
    probability: float
    time_elapsed: timedelta

Output = Union[NumericalOutput, CategoricalOutput]


@dataclass
class ShortDescription:
    name: str
    display: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "display": self.display
        }
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
        raise NotImplementedError("ModelRepository is deprecated")

@dataclass
class Metadata:
    Description: str
    Transformation: List[InputItem]
    Model: List[MetadataItem]

@dataclass
class ModelData:
    model: callable 
    model_manifest: List[MetadataItem]
    transformation: Dict[str, Any]
    transformation_manifest: List[InputItem]
    ordering: List[str] = field(default_factory=list)

    def _to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "description": self._find_metadata("description").value,
            "input": [tm.to_dict() for tm in self.transformation_manifest],
            "metadata": [mm.to_dict() for mm in self.model_manifest],
        }

    def _find_metadata(self, key: str) -> MetadataItem:
        for item in self.model_manifest:
            if item.key == key:
                return item
        return MetadataItem()

    def get_model_name(self) -> str:
        name = self._find_metadata("model_name")
        return name.value

    def get_short_description(self) -> ShortDescription:
        name = self._find_metadata("model_name")
        return ShortDescription(name=name.value, display="")

    def get_metadata(self) -> Dict[str, Any]:
        return self._to_metadata_dict()

    def _transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transformed_data = {}
        for key, value in data.items():
            if key not in self.transformation:
                transformed_data[key] = value; continue
            for _, transform_func in self.transformation[key].items():
                value = transform_func.transform(np.array(value).reshape(-1, 1))
            transformed_data[key] = value
        return transformed_data

    def _inverse_transform(self, result) -> any:
        for column, transformation_map in self.transformation.items():
            if column in [tm.key for tm in self.transformation_manifest]:
                continue
            for _, transform_func in transformation_map.items():
                result = transform_func.inverse_transform(result)
        return result

    def infer(self, data: Dict[str, Any]) -> Output:
        start_time = time.time()
        filtered_data = {key: value for key, value in data.items() if key in self._get_all_inputs()}
        transformed_data = self._transform(filtered_data)
        result = self.model.predict(self.dict_to_list(transformed_data))
        elapsed_time = time.time() - start_time
        result = self._inverse_transform(result)
        # TODO generalize for all model, not just numerical methods
        return NumericalOutput(value=result, time_elapsed=timedelta(seconds=elapsed_time))

    def dict_to_list(self, data: Dict[str, Any]) -> List[float]:
        if len(self.ordering) == 0:
            self.ordering = self._find_metadata("ordering").value
        arr = np.array([data[key] for key in self.ordering if key in data.keys()]).reshape(-1, 1)
        return arr

    def _get_all_inputs(self) -> List[str]:
        return [tm.key for tm in self.transformation_manifest]

    def validate_input(self, data: Dict[str, Any]) -> List[Input]:
        reconstruct = {}
        available_inputs = [tm.key for tm in self.transformation_manifest]
        input_keys = [key for key in data.keys()]
        for ai in available_inputs:
            if ai not in input_keys:
                return {}, f"key {ai} is missing"
            reconstruct[ai] = data[ai]
        return reconstruct, ""
    
class ModelServer:
    def __init__(self, repository: Repository):
        self.repository = repository
        self.available_models: List[ModelData] = []

    def load(self) -> "ModelData":
        parent_runs = self.repository.get_all_available_runs()
        for run in parent_runs:
            [models, model_manifests] = self.repository.load_all_models(run)
            [transformation, transformation_manifest] = self.repository.load_transformations(run)
            for model, model_manifest in zip(models, model_manifests):
                model_manifest = Manifest.read_model_manifest(model_manifest)
                self.available_models.append(ModelData(
                    model=model,
                    model_manifest=model_manifest,
                    transformation=transformation,
                    transformation_manifest=Manifest.read_transformation_manifest(transformation_manifest),
                ))
        return self

    def list(self) -> List[ShortDescription]:
        return [ am.get_short_description() for am in self.available_models]

    def _find_model(self, model_name: str) -> ModelData:
        for model in self.available_models:
            if model.get_model_name() == model_name:
                return model
        return ModelData()

    def metadata(self, model_name: str):
        model = self._find_model(model_name)
        if model is None:
            return None
        return model.get_metadata()

    def validate_input(self, model_name: str, data: Dict[str, Any]) -> List[Input]:
        for model in self.available_models:
            if model.get_model_name() == model_name:
                return model.validate_input(data)

    def infer(self, model_name: str, data: Dict[str, Any]) -> Output:
        model =  self._find_model(model_name)
        if model is None:
            return None
        return model.infer(data)

    