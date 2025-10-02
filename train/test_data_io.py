import pandas as pd
import pytest
import os
import warnings

# Suppress warnings for the entire module
warnings.filterwarnings("ignore")
# Or for specific warning types
# warnings.filterwarnings("ignore", category=UserWarning)

from .data_io import Disk
from enum import Enum

class SampleEnum(Enum):
    COLUMN_ID = 1
    COLUMN_FEATURE = 2
    COLUMN_TARGET = 3

FOLDER = "test"
FILENAME = "sample"

@pytest.fixture(scope="session")
def sample_csv_path():
    filepath = f"{FOLDER}/{FILENAME}.csv"
    csv_sample = """
    id,name,value
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
class TestDisk:
    """Test suite for disk-based data I/O operations using the Disk class."""

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_initialization(self):
        """Test Disk class initialization with folder path and filename."""
        disk = Disk(None, FOLDER, FILENAME)
        assert disk.path == FOLDER 
        assert disk.name == FILENAME

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_load_dataframe_via_csv(self, sample_csv_path):
        """Test loading CSV file into DataFrame with column name mapping using Enums."""
        disk = Disk(None, FOLDER, FILENAME)
        disk.load_dataframe_via_csv(SampleEnum, {})
        df = disk.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 3)
        assert list(df.columns) == [SampleEnum.COLUMN_ID.name, SampleEnum.COLUMN_FEATURE.name, SampleEnum.COLUMN_TARGET.name]
        assert df.iloc[0, 0] == 1
        assert df.iloc[0, 1] == 'hello'
        assert df.iloc[0, 2] == 300

    def test_replace_columns(self):
        """Test DataFrame column replacement using Enum mapping to standardize column names."""
        sample_dict = {
            "id": [1, 2, 3],
            "ffaster": ["asd", "asd", "bds"],
            "ttarget": [100, 200, 100],
        }
        disk = Disk(None, FOLDER, FILENAME)
        disk.raw_data = pd.DataFrame(sample_dict)
        disk.raw_data = disk._replace_columns(disk.raw_data, SampleEnum)
        
        assert disk.raw_data.columns[0] == SampleEnum.COLUMN_ID.name
        assert disk.raw_data.columns[1] == SampleEnum.COLUMN_FEATURE.name
        assert disk.raw_data.columns[2] == SampleEnum.COLUMN_TARGET.name

    def test_save_data_via_csv(self):
        """Test saving DataFrame to CSV file and verify the saved file contents."""
        output = "output"
        output_path = os.path.join(FOLDER, f"{output}.csv")
        sample_data = pd.DataFrame({
            "id": [1, 2], 
            "name": ["foo", "bar"], 
            "value": [100, 200]
        })
        
        disk = Disk(None, FOLDER, output)
        disk.save_via_csv()
        disk.save_data(sample_data)
        
        assert os.path.exists(output_path)
        loaded_data = pd.read_csv(output_path)
        assert loaded_data.shape == (2, 3)
        assert list(loaded_data.columns) == ["id", "name", "value"]

