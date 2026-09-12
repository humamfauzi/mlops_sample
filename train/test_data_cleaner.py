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
        SampleEnum.COLUMN_ID: [1, 2, 3, 4],
        SampleEnum.COLUMN_FEATURE_DELETED: ["q", "q", "q", "q"],
        SampleEnum.COLUMN_FEATURE: ["a", "a", np.nan, "b"],
        SampleEnum.COLUMN_TARGET: [123, 321, 122, 365],
    }
    df = pd.DataFrame(ddict)
    df.set_index(SampleEnum.COLUMN_ID, inplace=True)
    return df


@pytest.fixture(scope="function")
def df_numeric():
    """DataFrame with numeric and categorical columns for filter_rows tests."""
    ddict = {
        SampleEnum.COLUMN_ID: [1, 2, 3, 4, 5, 6],
        SampleEnum.COLUMN_FEATURE: ["a", "b", "c", "a", "b", "c"],
        SampleEnum.COLUMN_TARGET: [10, 20, 30, 40, 50, 60],
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
        cleaned_df = ddf.remove_columns([SampleEnum.COLUMN_FEATURE_DELETED]).clean_data(df)
        assert cleaned_df.shape == (4, 2)
        assert cleaned_df.columns[0] == SampleEnum.COLUMN_FEATURE
        assert cleaned_df.columns[1] == SampleEnum.COLUMN_TARGET

    def test_remove_nan_rows_lazy(self, df):
        ddf = Cleaner(None)
        ddf.column = SampleEnum  # set column enum
        cleaned_df = ddf.remove_nan_rows().clean_data(df)
        assert cleaned_df.shape == (3, 3)  # should remove one row with NaN

        with pytest.raises(
            KeyError
        ):  # should raise KeyError because index no longer exists
            cleaned_df.loc[3]

    def test_chain_operations_lazy(self, df):
        ddf = Cleaner(None)
        ddf.column = SampleEnum  # set column enum
        cleaned_df = (
            ddf.remove_columns([SampleEnum.COLUMN_FEATURE_DELETED])
            .remove_nan_rows()
            .clean_data(df)
        )
        assert cleaned_df.shape == (3, 2)  # one column and one row removed
        assert cleaned_df.columns[0] == SampleEnum.COLUMN_FEATURE
        assert cleaned_df.columns[1] == SampleEnum.COLUMN_TARGET

    def test_filter_columns_lazy(self, df):
        ddf = Cleaner(None)
        ddf.column = SampleEnum  # set column enum
        cleaned_df = ddf.filter_columns(
            [SampleEnum.COLUMN_FEATURE, SampleEnum.COLUMN_TARGET]
        ).clean_data(df)
        assert cleaned_df.shape == (4, 2)
        assert cleaned_df.columns[0] == SampleEnum.COLUMN_FEATURE
        assert cleaned_df.columns[1] == SampleEnum.COLUMN_TARGET

    def test_combined_operations_lazy(self, df):
        ddf = Cleaner(None)
        ddf.column = SampleEnum  # set column enum
        cleaned_df = (
            ddf.filter_columns(
                [
                    SampleEnum.COLUMN_FEATURE,
                    SampleEnum.COLUMN_TARGET,
                    SampleEnum.COLUMN_FEATURE_DELETED,
                ]
            )
            .remove_columns([SampleEnum.COLUMN_FEATURE_DELETED])
            .remove_nan_rows()
            .clean_data(df)
        )
        assert cleaned_df.shape == (3, 2)  # one column and one row removed
        assert cleaned_df.columns[0] == SampleEnum.COLUMN_FEATURE
        assert cleaned_df.columns[1] == SampleEnum.COLUMN_TARGET


class TestFilterRows:
    """Tests for the filter_rows lazy operation."""

    def test_filter_rows_in_operator(self, df_numeric):
        """Keep only rows where COLUMN_FEATURE is in a given list."""
        ddf = Cleaner(None)
        ddf.column = SampleEnum
        result = ddf.filter_rows(
            SampleEnum.COLUMN_FEATURE.name, "in", ["a", "b"]
        ).clean_data(df_numeric)
        assert len(result) == 4
        assert set(result[SampleEnum.COLUMN_FEATURE].unique()) == {"a", "b"}

    def test_filter_rows_not_in_operator(self, df_numeric):
        """Exclude rows where COLUMN_FEATURE is in a given list."""
        ddf = Cleaner(None)
        ddf.column = SampleEnum
        result = ddf.filter_rows(
            SampleEnum.COLUMN_FEATURE.name, "not_in", ["c"]
        ).clean_data(df_numeric)
        assert len(result) == 4
        assert "c" not in result[SampleEnum.COLUMN_FEATURE].values

    def test_filter_rows_eq_operator(self, df_numeric):
        """Keep rows where COLUMN_TARGET equals a specific value."""
        ddf = Cleaner(None)
        ddf.column = SampleEnum
        result = ddf.filter_rows(SampleEnum.COLUMN_TARGET.name, "eq", [30]).clean_data(
            df_numeric
        )
        assert len(result) == 1
        assert result[SampleEnum.COLUMN_TARGET].iloc[0] == 30

    def test_filter_rows_ne_operator(self, df_numeric):
        """Keep rows where COLUMN_TARGET does not equal a specific value."""
        ddf = Cleaner(None)
        ddf.column = SampleEnum
        result = ddf.filter_rows(SampleEnum.COLUMN_TARGET.name, "ne", [30]).clean_data(
            df_numeric
        )
        assert len(result) == 5
        assert 30 not in result[SampleEnum.COLUMN_TARGET].values

    def test_filter_rows_gt_operator(self, df_numeric):
        """Keep rows where COLUMN_TARGET > threshold."""
        ddf = Cleaner(None)
        ddf.column = SampleEnum
        result = ddf.filter_rows(SampleEnum.COLUMN_TARGET.name, "gt", [30]).clean_data(
            df_numeric
        )
        assert len(result) == 3
        assert all(result[SampleEnum.COLUMN_TARGET] > 30)

    def test_filter_rows_lt_operator(self, df_numeric):
        """Keep rows where COLUMN_TARGET < threshold."""
        ddf = Cleaner(None)
        ddf.column = SampleEnum
        result = ddf.filter_rows(SampleEnum.COLUMN_TARGET.name, "lt", [30]).clean_data(
            df_numeric
        )
        assert len(result) == 2
        assert all(result[SampleEnum.COLUMN_TARGET] < 30)

    def test_filter_rows_gte_operator(self, df_numeric):
        """Keep rows where COLUMN_TARGET >= threshold."""
        ddf = Cleaner(None)
        ddf.column = SampleEnum
        result = ddf.filter_rows(SampleEnum.COLUMN_TARGET.name, "gte", [30]).clean_data(
            df_numeric
        )
        assert len(result) == 4
        assert all(result[SampleEnum.COLUMN_TARGET] >= 30)

    def test_filter_rows_lte_operator(self, df_numeric):
        """Keep rows where COLUMN_TARGET <= threshold."""
        ddf = Cleaner(None)
        ddf.column = SampleEnum
        result = ddf.filter_rows(SampleEnum.COLUMN_TARGET.name, "lte", [30]).clean_data(
            df_numeric
        )
        assert len(result) == 3
        assert all(result[SampleEnum.COLUMN_TARGET] <= 30)

    def test_filter_rows_resets_index(self, df_numeric):
        """After filtering the index must be contiguous (reset_index applied)."""
        ddf = Cleaner(None)
        ddf.column = SampleEnum
        result = ddf.filter_rows(
            SampleEnum.COLUMN_FEATURE.name, "in", ["a"]
        ).clean_data(df_numeric)
        assert list(result.index) == list(range(len(result)))

    def test_filter_rows_chained_with_filter_columns(self, df_numeric):
        """filter_rows and filter_columns can be chained in any order."""
        ddf = Cleaner(None)
        ddf.column = SampleEnum
        result = (
            ddf.filter_rows(SampleEnum.COLUMN_FEATURE.name, "in", ["a", "b"])
            .filter_columns([SampleEnum.COLUMN_FEATURE, SampleEnum.COLUMN_TARGET])
            .clean_data(df_numeric)
        )
        assert result.shape == (4, 2)
        assert SampleEnum.COLUMN_FEATURE in result.columns
        assert SampleEnum.COLUMN_TARGET in result.columns

    def test_filter_rows_unsupported_operator_raises(self, df_numeric):
        """An unsupported operator must raise ValueError at queue time."""
        ddf = Cleaner(None)
        ddf.column = SampleEnum
        with pytest.raises(ValueError, match="unsupported operator"):
            ddf.filter_rows(SampleEnum.COLUMN_FEATURE.name, "contains", ["a"])

    def test_filter_rows_scalar_operator_multiple_values_raises(self, df_numeric):
        """Scalar operators (eq, gt, …) must raise ValueError when given multiple values."""
        ddf = Cleaner(None)
        ddf.column = SampleEnum
        with pytest.raises(ValueError, match="exactly one value"):
            ddf.filter_rows(SampleEnum.COLUMN_TARGET.name, "gt", [10, 20])

    def test_filter_rows_parse_instruction(self):
        """parse_instruction correctly wires up a filter_rows step from a config dict."""
        config_call = [
            {
                "type": "filter_rows",
                "column": "column_feature",
                "operator": "in",
                "values": ["a", "b"],
            },
        ]
        ddf = Cleaner.parse_instruction({"reference": "sample"}, config_call, None)
        # should have exactly one queued operation
        assert len(ddf.call_container) == 1

    def test_drop_na_alias_in_parse_instruction(self):
        """'drop_na' in config is treated identically to 'remove_nan_rows'."""
        config_call = [{"type": "drop_na"}]
        ddf = Cleaner.parse_instruction({"reference": "sample"}, config_call, None)
        assert len(ddf.call_container) == 1


class TestOperationKinds:
    """Queued operations are tagged, so a scorer can skip population filters.

    `filter_rows` decides which rows are in scope; `drop_na` keeps the frame
    usable; column selection shapes it. Only the first should be skippable when
    re-scoring a trimmed model on the full population.
    """

    def _cleaner(self):
        cleaner = Cleaner(None)
        cleaner.column = SampleEnum
        return cleaner

    def test_filter_rows_is_a_population_filter(self):
        c = self._cleaner()
        c.filter_rows(SampleEnum.COLUMN_FEATURE.name, "eq", ["a"])

        assert c.call_container[0][0] == Cleaner.POPULATION

    def test_drop_na_is_hygiene(self):
        c = self._cleaner()
        c.remove_nan_rows()

        assert c.call_container[0][0] == Cleaner.HYGIENE

    def test_column_operations_are_neither(self):
        c = self._cleaner()
        c.filter_columns([SampleEnum.COLUMN_FEATURE.name])
        c.remove_columns([SampleEnum.COLUMN_FEATURE.name])

        assert [kind for kind, _ in c.call_container] == [Cleaner.COLUMNS] * 2

    def test_skipping_population_filters_keeps_hygiene_and_columns(self, df_numeric):
        c = self._cleaner()
        c.filter_columns([SampleEnum.COLUMN_FEATURE, SampleEnum.COLUMN_TARGET])
        c.filter_rows(SampleEnum.COLUMN_TARGET.name, "gt", [1000])   # matches nothing
        c.remove_nan_rows()

        with_filters = c.clean_data(df_numeric)
        without = c.clean_data(df_numeric, apply_population_filters=False)

        assert len(with_filters) == 0                    # filter_rows applied
        assert len(without) == len(df_numeric)           # skipped
        assert list(without.columns) == [SampleEnum.COLUMN_FEATURE,
                                         SampleEnum.COLUMN_TARGET]

    def test_record_metadata_false_writes_nothing(self, df_numeric):
        class Facade:
            def __init__(self):
                self.calls = []

            def set_data_cleaning_time(self, ms):
                self.calls.append("time")

            def set_row_size_after_cleaning(self, n):
                self.calls.append("rows")

            def set_column_size_after_cleaning(self, n):
                self.calls.append("cols")

        c = self._cleaner()
        c.facade = Facade()

        c.clean_data(df_numeric, record_metadata=False)
        assert c.facade.calls == []

        c.clean_data(df_numeric, record_metadata=True)
        assert c.facade.calls == ["time", "rows", "cols"]
