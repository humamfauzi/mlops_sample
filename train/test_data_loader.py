import pandas as pd
import pytest
import os

from .data_loader import Disk
from enum import Enum

class SampleEnum(Enum):
    COLUMN_ID = 1
    COLUMN_FEATURE = 2
    COLUMN_TARGET = 3


# it will initate write a dummy csv to a disk for initialization
# and it will remove it once all the test is done
@pytest.fixture(scope="session")
def path():
    path = "sample.csv"
    csv_sample = """
    id,name,value
    1,hello,300
    """
    with open(path, 'w') as f: 
        f.write(csv_sample)
    yield path
    if os.path.exists(path):
        os.remove(path)
    
class TestDisk:
    """
    Test suite for the Disk class functionality.
    
    This class contains tests to verify the column replacement functionality
    of the Disk class using Enum-based column mapping.
    """

    def test_load_data(self, path):
        disk = Disk(path, SampleEnum)
        df = disk.load_data()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1,3)

    def test_replace_columns(self):
        """
        Test the column replacement functionality of the Disk class.
        
        This test verifies that the replace_columns method correctly maps
        the original column names to their corresponding Enum values.
        
        Test Steps:
        1. Create a sample dictionary with test data
        2. Initialize a Disk instance with a sample path and Enum
        3. Create a DataFrame from the sample dictionary
        4. Call replace_columns method
        5. Verify column names are correctly replaced with Enum values
        
        Expected Results:
        - First column should be mapped to SampleEnum.COLUMN_ID
        - Second column should be mapped to SampleEnum.COLUMN_FEATURE
        - Third column should be mapped to SampleEnum.COLUMN_TARGET
        
        Raises:
        -------
        AssertionError
            If any column length is not equal enum length
        """
        sample_dict = {
            "id": [1,2,3],
            "ffaster": ["asd", "asd", "bds"],
            "ttarget": [100, 200, 100],
        }
        disk = Disk("/sample", SampleEnum)
        disk.raw_data = pd.DataFrame(sample_dict)
        disk.replace_columns()
        
        assert disk.raw_data.columns[0] == SampleEnum.COLUMN_ID
        assert disk.raw_data.columns[1] == SampleEnum.COLUMN_FEATURE
        assert disk.raw_data.columns[2] == SampleEnum.COLUMN_TARGET

        sample_dict = {}
        disk = Disk("/sample", SampleEnum)
        disk.raw_data = pd.DataFrame(sample_dict)
        with pytest.raises(ValueError):
            disk.replace_columns()
