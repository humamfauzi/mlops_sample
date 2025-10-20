import pytest
import pandas as pd
import numpy as np

from copy import copy
from train.data_cleaner import Cleaner
from column.cfs2017 import SampleEnum

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

class TestDataFrameLazyCall:
    """
    Test suite for the DataFrameLazyCall class of cleaning
    It assumes that all column already converted to enums
    """

    def test_remove_column_lazy(self, df):
        ddf = Cleaner(None)
        ddf.column = SampleEnum  # set column enum
        cleaned_df = ddf.remove_columns([SampleEnum.COLUMN_REMOVED]).clean_data(df)
        assert cleaned_df.shape == (4, 2)
        assert cleaned_df.columns[0] == SampleEnum.COLUMN_FEATURE
        assert cleaned_df.columns[1] == SampleEnum.COLUMN_TARGET

    def test_remove_nan_rows_lazy(self, df):
        ddf = Cleaner(None)
        ddf.column = SampleEnum  # set column enum
        cleaned_df = ddf.remove_nan_rows().clean_data(df)
        assert cleaned_df.shape == (3, 3)  # should remove one row with NaN

        with pytest.raises(KeyError):  # should raise KeyError because index no longer exists
            cleaned_df.loc[3]

    def test_chain_operations_lazy(self, df):
        ddf = Cleaner(None)
        ddf.column = SampleEnum  # set column enum
        cleaned_df = (
            ddf
            .remove_columns([SampleEnum.COLUMN_REMOVED])
            .remove_nan_rows()
            .clean_data(df)
        )
        assert cleaned_df.shape == (3, 2)  # one column and one row removed
        assert cleaned_df.columns[0] == SampleEnum.COLUMN_FEATURE
        assert cleaned_df.columns[1] == SampleEnum.COLUMN_TARGET

    def test_filter_columns_lazy(self, df):
        ddf = Cleaner(None)
        ddf.column = SampleEnum  # set column enum
        cleaned_df = ddf.filter_columns([SampleEnum.COLUMN_FEATURE, SampleEnum.COLUMN_TARGET]).clean_data(df)
        assert cleaned_df.shape == (4, 2)
        assert cleaned_df.columns[0] == SampleEnum.COLUMN_FEATURE
        assert cleaned_df.columns[1] == SampleEnum.COLUMN_TARGET

    def test_combined_operations_lazy(self, df):
        ddf = Cleaner(None)
        ddf.column = SampleEnum  # set column enum
        cleaned_df = (
            ddf
            .filter_columns([SampleEnum.COLUMN_FEATURE, SampleEnum.COLUMN_TARGET, SampleEnum.COLUMN_REMOVED])
            .remove_columns([SampleEnum.COLUMN_REMOVED])
            .remove_nan_rows()
            .clean_data(df)
        )
        assert cleaned_df.shape == (3, 2)  # one column and one row removed
        assert cleaned_df.columns[0] == SampleEnum.COLUMN_FEATURE
        assert cleaned_df.columns[1] == SampleEnum.COLUMN_TARGET