import pytest
import os

from train.scenario_manager import InstructionFactory, ScenarioManager
from train.sstruct import Pairs, Stage, FeatureTargetPair
import pandas as pd

FOLDER = "test"
FILENAME = "sample"

@pytest.fixture(scope="function")
def sample_csv_path():
    filepath = f"{FOLDER}/{FILENAME}.csv"
    csv_sample = """
    id,feature,feature_deleted,target
    1,123,del,300
    2,456,del,600
    3,789,del,900
    4,101112,del,1200
    5,131415,del,1500
    6,161718,del,1800
    7,192021,del,2100
    8,222324,del,2400
    9,252627,del,2700
    10,282930,del,3000
    """
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER) 
    with open(filepath, 'w') as f: 
        f.write(csv_sample)
    yield filepath
    if os.path.exists(filepath):
        os.remove(filepath)

@pytest.mark.filterwarnings("ignore")
class TestScenarioManager:
    """Test suite for scenario manager operations."""

    def test_data_disk(self, sample_csv_path):
        base = {
            "name": "base_test",
            "description": "A basic test to load data from disk",
            "repository": {},
            "instructions": [
                {
                    "type": "data_io",
                    "properties": {
                        "path": FOLDER,
                        "file": FILENAME,
                        "format": "csv",
                        "reference": "sample"
                    },
                    "call": [
                        {
                            "type": "load",
                            "n_rows": 1
                        }
                    ]
                }
            ]
        }
        instruction = InstructionFactory.parse_instruction(base)
        sm = ScenarioManager(instruction)
        assert sm is not None

        result = sm.construct().execute()
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 4)

    def test_data_disk_fail(self):
        base = {
            "name": "base_test",
            "description": "A basic test to load data from disk",
            "repository": {},
            "instructions": [
                {
                    "type": "data_io",
                    "properties": {
                        "path": FOLDER + "not_exist",
                        "file": FILENAME,
                        "format": "csv",
                        "reference": "sample"
                    },
                    "call": [
                        {
                            "type": "load",
                            "n_rows": 1
                        }
                    ]
                }
            ]
        }
        instruction = InstructionFactory.parse_instruction(base)
        sm = ScenarioManager(instruction)
        assert sm is not None

        with pytest.raises(FileNotFoundError):
            sm.construct().execute()

    def test_data_disk_clean(self,sample_csv_path):
        base = {
            "name": "base_test",
            "description": "A basic test to load data from disk and clean it",
            "repository": {},
            "instructions": [
                {
                    "type": "data_io",
                    "properties": {
                        "path": FOLDER,
                        "file": FILENAME,
                        "format": "csv",
                        "reference": "sample"
                    },
                    "call": [
                        {
                            "type": "load",
                        }
                    ]
                },
                {
                    "type": "data_cleaner",
                    "properties": {
                        "reference": "sample"
                    },
                    "call": [
                        {
                            "type": "filter_columns",
                            "columns": ["column_feature", "column_target"]
                        }
                    ]
                }
            ]
        }
        instruction = InstructionFactory.parse_instruction(base)
        sm = ScenarioManager(instruction)
        assert sm is not None

        result = sm.construct().execute()
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 2)

    def test_data_disk_clean_transform(self, sample_csv_path):
        base = {
            "name": "base_test",
            "description": "A basic test to load data from disk and clean it",
            "repository": {},
            "instructions": [
                {
                    "type": "data_io",
                    "properties": {
                        "path": FOLDER,
                        "file": FILENAME,
                        "format": "csv",
                        "reference": "sample"
                    },
                    "call": [
                        {
                            "type": "load",
                        }
                    ]
                },
                {
                    "type": "data_cleaner",
                    "properties": {
                        "reference": "sample"
                    },
                    "call": [
                        {
                            "type": "filter_columns",
                            "columns": ["column_feature", "column_target"]
                        }
                    ]
                },
                {
                    "type": "data_transformer",
                    "properties": {
                        "reference": "sample"
                    },
                    "call": [
                        {
                            "type": "log_transformation",
                            "condition": "replace",
                            "columns": ["column_feature", "column_target"]
                        },
                        {
                            "type": "normalization",
                            "condition": "replace",
                            "columns": ["column_feature", "column_target"]
                        }
                    ]
                }
            ]
        }
        instruction = InstructionFactory.parse_instruction(base)
        sm = ScenarioManager(instruction)
        assert sm is not None

        result = sm.construct().execute()
        assert result is not None
        assert isinstance(result, Pairs)
        assert isinstance(result.train, FeatureTargetPair)
        assert result.train.X.shape == (8, 2)

    def test_data_disk_clean_transform_train(self, sample_csv_path):
        base = {
            "name": "base_test",
            "description": "A basic test to load data from disk and clean it",
            "repository": {},
            "instructions": [
                {
                    "type": "data_io",
                    "properties": {
                        "path": FOLDER,
                        "file": FILENAME,
                        "format": "csv",
                        "reference": "sample"
                    },
                    "call": [
                        {
                            "type": "load",
                        }
                    ]
                },
                {
                    "type": "data_cleaner",
                    "properties": {
                        "reference": "sample"
                    },
                    "call": [
                        {
                            "type": "filter_columns",
                            "columns": ["column_feature", "column_target"]
                        }
                    ]
                },
                {
                    "type": "data_transformer",
                    "properties": {
                        "reference": "sample"
                    },
                    "call": [
                        {
                            "type": "log_transformation",
                            "condition": "replace",
                            "columns": ["column_feature", "column_target"]
                        },
                        {
                            "type": "normalization",
                            "condition": "replace",
                            "columns": ["column_feature", "column_target"]
                        }
                    ]
                },
                {
                    "type": "model_trainer",
                    "properties": {
                        "objective": "first_model",
                        "random_state": 42,
                        "fold": 5,
                        "parameter_grid": {
                            "type": "exhaustive"
                        },
                        "metrics": ["rmse"],
                        "primary_metric": "rmse"
                    },
                    "call": [
                        {
                            "model_type": "random_forest_regressor",
                            "hyperparameters": {
                                "n_estimators": [50, 100],
                                "max_depth": [5, 10]
                            }
                        },
                        {
                            "model_type": "linear_regression",
                            "hyperparameters": {}
                        }
                    ]
                }
            ]
        }
        instruction = InstructionFactory.parse_instruction(base)
        sm = ScenarioManager(instruction)
        assert sm is not None

        result = sm.construct().execute()
        assert result is not None