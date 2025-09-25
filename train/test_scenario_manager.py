import pytest
import os

from train.scenario_manager import InstructionFactory, ScenarioManager
import pandas as pd

FOLDER = "test"
FILENAME = "sample"

@pytest.fixture(scope="session")
def sample_csv_path():
    filepath = f"{FOLDER}/{FILENAME}.csv"
    csv_sample = """
    id,feature,target
    1,hello,300
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
        assert result.shape == (1, 3)


