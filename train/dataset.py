from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from typing import Optional, Generic, TypeVar, Dict, List
from dataclasses import dataclass, field, asdict
import numpy as np
import json

@dataclass
class TabularDatasetMetadata:
    count: int = 0
    size: int = 0

@dataclass
class ColumnNumericalProperties:
    mean: float = 0
    count: int = 0 
    stddev: float = 0
    sum: float = 0
    min: float = 0
    max: float = 0 

@dataclass
class ColumnCategoricalProperties:
    unique: List[str]
    count: int
    freq: Dict[str, int]

ColumnNames = TypeVar('ColumnNames')
@dataclass
class ColumnProperties(Generic[ColumnNames]):
    numerical: Dict[ColumnNames, ColumnNumericalProperties] = field(default_factory=dict)
    categorical: Dict[ColumnNames, ColumnCategoricalProperties] = field(default_factory=dict)

@dataclass
class TabularDatasetProperties:
    source: str = field(default="")
    name: str = field(default="")
    description: str = field(default="")
    digest: str = field(default="")
    source_type: str = field(default="")
    meta: Optional[TabularDatasetMetadata] = None
    columns: Optional[ColumnProperties] = None
    schema: Optional[Dict[str, str]] = None



class TrackingDataset(Dataset):
    def __init__(self, source: str, name: str, description: str):
        self.properties: TabularDatasetProperties = TabularDatasetProperties(
            source=source,
            name=name,
            description=description,
            digest="ASD",
            meta=None,
            columns=None
        )
        return

    def name(self):
        return self.properties.name

    def digest(self):
        return "DIGEST"

    def source(self):
        return self.properties.source

    def to_dict(self):
        dprofile = {}
        if self.properties.meta is not None:
            dprofile["meta"] = asdict(self.properties.meta)
        if self.properties.columns is not None:
            dprofile["columns"] = asdict(self.properties.columns)
        profile = dict_to_json_bin(dprofile)
        ddict = {
            "name": self.properties.name,
            "digest": self.properties.digest,
            "source_type": self.properties.source_type,
            "source": self.properties.source,
            "schema": self.properties.schema,
            "profile": str(profile),
        }
        return ddict

    def to_json(self):
        return json.dumps(self.properties)

    def set_source(self, source: str):
        self.properties.source = source
        return self

    def set_description(self, description: str):
        self.properties.description = description
        return self

    def set_name(self, name: str):
        self.properties.name = name
        return self
    
    def set_metadata(self, count: int, size: int):
        self.properties.meta = TabularDatasetMetadata(count, size)
        return self

    def set_numerical_column_properties(self, column: ColumnNames, mean: float, count: int, stddev: float, sum: float, min: float, max: float):
        if self.properties.columns is None:
            self.properties.columns = ColumnProperties(numerical={}, categorical={})
        self.properties.columns.numerical[column] = ColumnNumericalProperties(mean, count, stddev, sum, min, max)
        return self

    def set_categorical_column_properties(self, column: ColumnNames, unique: List[str], count: int, freq: Dict[str, int]):
        if self.properties.columns is None:
            self.properties.columns = ColumnProperties(numerical={}, categorical={})
        self.properties.columns.categorical[column] = ColumnCategoricalProperties(unique, count, freq)
        return self

    def mlflow(self):
        return self.properties

    def to_proto():
        return ""

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def dict_to_json_bin(dictionary):
    json_str = json.dumps(dictionary, cls=NumpyEncoder)
    json_bin = json_str.encode('utf-8')
    return json_bin