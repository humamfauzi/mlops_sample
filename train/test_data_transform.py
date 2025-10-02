import pytest
import numpy as np
import pandas as pd
import pickle
import os

from enum import Enum
from train.data_transform import Transformer, TransformationMethods, Keeper
from train.column import SampleEnumTransformer as SampleEnum


TRACKING_PATH = "local"
EXPERIMENT_NAME = "test"

PROPS = { "reference": "sample_enum_transformer",}
@pytest.fixture(scope='function')
def df():
    ddict = {
        SampleEnum.COLUMN_ID.name: np.arange(0, 10),
        SampleEnum.COLUMN_CATEGORICAL.name: ['a', 'b'] * 5,
        SampleEnum.COLUMN_NUMERICAL.name: np.arange(10, 20),
        SampleEnum.COLUMN_TARGET.name: np.random.random(10),
    }
    df = pd.DataFrame(ddict)
    df.set_index(SampleEnum.COLUMN_ID.name, inplace=True)
    return df

class TestDataTransformLazyCall:
    def test_log_transformation(self, df):
        log_dict = {
            "type": "log_transformation",
            "columns": ["column_numerical"],
            "condition": "replace"
        }
        dtlc = Transformer.parse_instruction(PROPS, [log_dict], None)
        pairs = dtlc.execute(df)
        # pick one sample
        row = pairs.train.X.iloc[0]
        index, num_val = row.name, row[SampleEnum.COLUMN_NUMERICAL.name]
        assert num_val == np.log(df.loc[index][SampleEnum.COLUMN_NUMERICAL.name])

    def test_min_max_transformation(self, df):
        minmax_dict = {
            "type": "min_max_transformation",
            "columns": ["column_numerical"],
            "condition": "replace"
        }
        dtlc = Transformer.parse_instruction(PROPS, [minmax_dict], None)
        pairs = dtlc.execute(df)
        assert pairs.train.X.loc[0][SampleEnum.COLUMN_NUMERICAL.name] == 0

    def test_add_one_hot_encoding_transformation(self, df):
        ohe_dict = {
            "type": "one_hot_encoding",
            "columns": ["column_categorical"],
            "condition": "append_and_remove"
        }
        dtlc = Transformer.parse_instruction(PROPS, [ohe_dict], None)
        pairs = dtlc.execute(df)
        assert pairs.train.X.shape == (8, 3)
        assert pairs.valid.X.shape == (1, 3)
        assert pairs.test.X.shape == (1, 3)
        # while it seems random, it is not. because when splitting we set the random seed
        # therefore any test picking index 0 should always be the same
        assert pairs.train.X.loc[0][SampleEnum.COLUMN_CATEGORICAL.name + '_b'] == 0