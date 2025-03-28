import pytest
import numpy as np
import pandas as pd
import pickle
import os

from enum import Enum
from train.data_transform import DataTransform, DataTransformLazyCall, TransformationMethods
from repositories.dummy import DummyMLflowRepository
from train.sstruct import Stage, FeatureTargetPair


TRACKING_PATH = "local"
EXPERIMENT_NAME = "test"
class SampleEnum(Enum):
    COLUMN_ID = 1
    COLUMN_CATEGORICAL = 2
    COLUMN_NUMERICAL = 3
    COLUMN_TARGET = 4

    @classmethod
    def categorical(cls):
        return [cls.COLUMN_CATEGORICAL]

    @classmethod
    def numerical(cls):
        return [cls.COLUMN_NUMERICAL]

    @classmethod
    def feature(cls, current_column):
        alll = cls.numerical() + cls.categorical()
        return list(set(alll) & set(current_column))

    @classmethod
    def target(cls):
        return cls.COLUMN_TARGET

@pytest.fixture(scope='function')
def df():
    ddict = {
        SampleEnum.COLUMN_ID: np.arange(0, 10),
        SampleEnum.COLUMN_CATEGORICAL: ['a', 'b'] * 5,
        SampleEnum.COLUMN_NUMERICAL: np.arange(10, 20),
        SampleEnum.COLUMN_TARGET: np.random.random(10),
    }
    df = pd.DataFrame(ddict)
    df.set_index(SampleEnum.COLUMN_ID, inplace=True)
    return df

@pytest.fixture(scope='function')
def dummy():
    return DummyMLflowRepository(TRACKING_PATH, EXPERIMENT_NAME)

class TestDataTransform:
    # TODO there are several warning beacause data frame copy,
    # we need to fix it after all test unit and CI/CD established
    def test_split_page(self, df):
        pass
        dt = DataTransform(SampleEnum)
        dt.transformed_data = df
        dt.split_stage()

        # every split would be .8 .1 .1 for train, valid, and test respectively
        # it should have two because id is an index so does not count, and target
        # already separated in a y
        assert dt.train_pair.X.shape == (8, 2)
        assert dt.valid_pair.X.shape == (1, 2)
        assert dt.test_pair.X.shape == (1, 2)

        assert dt.train_pair.y.shape == (8,)
        assert dt.valid_pair.y.shape == (1,)
        assert dt.test_pair.y.shape == (1,)

    def test_one_hot_encoding(self, df):
        pass
        dt = DataTransform(SampleEnum)
        dt.train_pair = FeatureTargetPair(
            df[SampleEnum.feature(df.columns)], 
            df[SampleEnum.target()],
            Stage.TRAIN,
        )
        dt.one_hot_encoding()

        # it should be 3 because the categorical have two value a and b
        # it would drop the original column when finish expanding
        assert dt.train_pair.X.shape == (10, 3)

    def test_min_max(self, df):
        pass
        dt = DataTransform(SampleEnum)
        dt.train_pair = FeatureTargetPair(
                df[SampleEnum.feature(df.columns)],
                df[SampleEnum.target()],
                Stage.TRAIN)
        dt.minmax()

        # it should stay 2 because min max does not alter column
        assert dt.train_pair.X.shape == (10, 2)
        assert dt.train_pair.X.loc[0][SampleEnum.COLUMN_NUMERICAL] == 0
        assert dt.train_pair.X.loc[9][SampleEnum.COLUMN_NUMERICAL] == 1

    def test_transformation_object(self, df):
        pass
        """
        this should prove that transformation saved as pickle object able
        to be loaded and tranform incoming data. This is important for
        inference.
        """
        dt = DataTransform(SampleEnum)
        dt.train_pair = FeatureTargetPair(
            df[SampleEnum.feature(df.columns)], 
            df[SampleEnum.target()],
            Stage.TRAIN,
        )
        dt.one_hot_encoding()

        obj = dt.ohe[SampleEnum.COLUMN_CATEGORICAL]
        # stored in artifacts because it already incuded in .gitignore
        dt.save_preprocessing_as_object("artifacts", "ohe", SampleEnum.COLUMN_CATEGORICAL, obj)
        with open(f"artifacts/{SampleEnum.COLUMN_CATEGORICAL}/ohe.pkl", 'rb') as file:
            ohe_func = pickle.load(file)
            result = ohe_func.transform(np.array(['a']).reshape(1, -1))

            # should expanded to two columns because ohe transformation
            assert result.shape == (1, 2)
            # the first one is represent 'a' categories so it should equal to 1
            assert result[0][0] == 1
            # the second one is represent 'b' categories so it should equal to 0
            assert result[0][1] == 0

    def test_reapply(self, df):
        pass
        dt = DataTransform(SampleEnum)
        dt.transformed_data = df
        dt.split_stage()
        assert dt.train_pair.X.shape == (8, 2)
        assert dt.valid_pair.X.shape == (1, 2)
        assert dt.test_pair.X.shape == (1, 2)

        dt.one_hot_encoding().reapply()

        assert dt.train_pair.X.shape == (8, 3)
        assert dt.valid_pair.X.shape == (1, 3)
        assert dt.test_pair.X.shape == (1, 3)

class TestDataTransformLazyCall:
    def test_log_transformation(self, df, dummy):
        dtlc = DataTransformLazyCall(SampleEnum, dummy)
        dtlc.add_log_transformation(SampleEnum.COLUMN_NUMERICAL, TransformationMethods.REPLACE)
        pairs = dtlc.transform_data(df)
        # pick one sample
        row = pairs.train.X.iloc[0]
        index, num_val = row.name, row[SampleEnum.COLUMN_NUMERICAL]
        assert num_val == np.log(df.loc[index][SampleEnum.COLUMN_NUMERICAL])

    def test_min_max_transformation(self, df, dummy):
        dtlc = DataTransformLazyCall(SampleEnum, dummy)
        dtlc.add_min_max_transformation(SampleEnum.COLUMN_NUMERICAL, TransformationMethods.REPLACE)
        pairs = dtlc.transform_data(df)
        assert pairs.train.X.loc[0][SampleEnum.COLUMN_NUMERICAL] == 0

    def test_add_one_hot_encoding_transformation(self, df, dummy):
        dtlc = DataTransformLazyCall(SampleEnum, dummy)
        dtlc.add_one_hot_encoding_transformation(SampleEnum.COLUMN_CATEGORICAL)
        pairs = dtlc.transform_data(df)
        assert pairs.train.X.shape == (8, 3)
        assert pairs.valid.X.shape == (1, 3)
        assert pairs.test.X.shape == (1, 3)
        # while it seems random, it is not. because when splitting we set the random seed
        # therefore any test picking index 0 should always be the same
        assert pairs.train.X.loc[0][SampleEnum.COLUMN_CATEGORICAL.name + '_b'] == 1