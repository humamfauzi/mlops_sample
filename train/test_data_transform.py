import pytest
import numpy as np
import pandas as pd
import json
import pickle
import os

from enum import Enum
from train.data_transform import Transformer, TransformationMethods, Keeper
from column.cfs2017 import CommodityFlow
from column.cfs2017 import SampleEnumTransformer as SampleEnum
from column.cfs2017 import SampleEnum as FixtureEnum


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


class RecordingFacade:
    """Captures what _save_manifest would have written to the repository."""

    def __init__(self):
        self.objects = {}

    def set_object_transformation(self, key, value):
        self.objects[key] = value
        return self

    def manifest(self):
        return json.loads(self.objects["transformation.allowed_columns"])


class TestManifest:
    """F-08: the manifest decides what the server will accept as input.

    A column the manifest omits is a column the server rejects, so anything
    dropped here has to be dropped deliberately.
    """

    def test_classified_columns_all_appear(self, df):
        facade = RecordingFacade()
        transformer = Transformer(facade, SampleEnum)

        transformer._save_manifest(df)

        names = [c["name"] for c in facade.manifest()]
        assert SampleEnum.COLUMN_NUMERICAL.name in names
        assert SampleEnum.COLUMN_CATEGORICAL.name in names

    def test_target_and_primary_id_are_excluded(self, df):
        facade = RecordingFacade()
        transformer = Transformer(facade, SampleEnum)

        transformer._save_manifest(df)

        names = [c["name"] for c in facade.manifest()]
        assert SampleEnum.COLUMN_TARGET.name not in names
        assert SampleEnum.COLUMN_ID.name not in names

    def test_numerical_carries_the_training_range(self, df):
        facade = RecordingFacade()
        transformer = Transformer(facade, SampleEnum)

        transformer._save_manifest(df)

        entry = next(c for c in facade.manifest() if c["name"] == SampleEnum.COLUMN_NUMERICAL.name)
        assert entry["type"] == "numerical"
        assert (entry["min"], entry["max"]) == (10, 19)

    def test_categorical_carries_the_observed_values(self, df):
        facade = RecordingFacade()
        transformer = Transformer(facade, SampleEnum)

        transformer._save_manifest(df)

        entry = next(c for c in facade.manifest() if c["name"] == SampleEnum.COLUMN_CATEGORICAL.name)
        assert entry["type"] == "categorical"
        assert sorted(entry["available_values"]) == ["a", "b"]

    def test_unclassified_column_is_rejected(self, df):
        # SampleEnum.COLUMN_FEATURE_DELETED is in neither numerical() nor
        # categorical(). Previously it was dropped without a word, producing a
        # model the server could never feed -- the failure only surfaced at
        # inference time.
        frame = df.copy()
        frame[FixtureEnum.COLUMN_FEATURE_DELETED.name] = "del"
        facade = RecordingFacade()

        with pytest.raises(ValueError, match="neither numerical nor categorical"):
            Transformer(facade, FixtureEnum)._save_manifest(frame)

    def test_error_names_the_offending_columns(self, df):
        frame = df.copy()
        frame[FixtureEnum.COLUMN_FEATURE_DELETED.name] = "del"
        frame["SOME_OTHER_COLUMN"] = 1

        with pytest.raises(ValueError) as excinfo:
            Transformer(RecordingFacade(), FixtureEnum)._save_manifest(frame)

        message = str(excinfo.value)
        assert FixtureEnum.COLUMN_FEATURE_DELETED.name in message
        assert "SOME_OTHER_COLUMN" in message

    def test_no_facade_is_a_no_op(self, df):
        # The transform tests run without a repository; that must stay silent.
        assert Transformer(None, SampleEnum)._save_manifest(df) is not None


class TestCommodityFlowClassification:
    def test_boolean_flags_are_categorical(self):
        # These were in neither list, so any config using them trained on a
        # feature the manifest would silently refuse to expose.
        categorical = CommodityFlow.categorical()

        assert CommodityFlow.IS_EXPORT.name in categorical
        assert CommodityFlow.IS_TEMPERATURE_CONTROLLED.name in categorical

    def test_every_column_is_classified_somewhere(self):
        classified = (
            set(CommodityFlow.numerical())
            | set(CommodityFlow.categorical())
            | {CommodityFlow.target(), CommodityFlow.primary_id()}
        )
        unclassified = [m.name for m in CommodityFlow if m.name not in classified]

        assert unclassified == []

    def test_target_is_not_a_feature(self):
        # Guard the leak that the duplicated copy of this module had introduced.
        assert CommodityFlow.target() not in CommodityFlow.feature(
            [m.name for m in CommodityFlow]
        )


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