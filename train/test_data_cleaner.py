import pytest
import pandas as pd
import numpy as np

from copy import copy
from enum import Enum
from train.data_cleaner import DataFrame

class SampleEnum(Enum):
    COLUMN_ID = 1
    COLUMN_FEATURE = 2
    COLUMN_TARGET = 3
    COLUMN_REMOVED = 4

# it means each function call it would recreate the dataframe
@pytest.fixture(scope="function")
def df():
    ddict = {
        SampleEnum.COLUMN_ID: [1,2,3,4],
        SampleEnum.COLUMN_REMOVED: ['q', 'q', 'q', 'q'],
        SampleEnum.COLUMN_FEATURE: ["a", "a", np.nan, "b"],
        SampleEnum.COLUMN_TARGET: [123, 321, 122, 365],
    }
    df = pd.DataFrame(ddict)
    df.set_index(SampleEnum.COLUMN_ID, inplace=True)
    return df

class TestDataFrame:
    """
    Test suite for the data frame class of cleaning
    It assumes that all column already converted to enums

    No need to test the interface clean_data because
    it composed by many methods that returns dataframe
    we only need to test that many methods works as expected
    """
    def test_remove_column(self, df):
        ddf = DataFrame()
        ddf.cleaned_data = df # manually assign for testing purposes
        assert ddf.cleaned_data.shape == (4, 3)

        ddf.remove_columns([SampleEnum.COLUMN_REMOVED])
        assert ddf.cleaned_data.shape == (4, 2)
        # assert ddf.cleaned_data.columns[0] == SampleEnum.COLUMN_ID
        assert ddf.cleaned_data.columns[0] == SampleEnum.COLUMN_FEATURE
        assert ddf.cleaned_data.columns[1] == SampleEnum.COLUMN_TARGET

    def test_remove_nan_rows(self, df):
        ddf = DataFrame()
        ddf.cleaned_data = df # manually assign for testing purposes
        assert ddf.cleaned_data.shape == (4,3)
        
        ddf.remove_nan_rows()
        assert ddf.cleaned_data.shape == (3, 3) # should remove one value for nil

        with pytest.raises(KeyError): # should raise key error because index no longer exist
            ddf.cleaned_data.loc[3]

