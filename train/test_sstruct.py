import pytest
import numpy as np
import pandas as pd
from .sstruct import Stage, FeatureTargetPair

class TestStage:
    def test_name(self):
        assert Stage.TRAIN.name == "TRAIN"
        assert Stage.VALID.name == "VALID"
        assert Stage.TEST.name == "TEST"

    def test_value(self):
        assert Stage.TRAIN.value == 1
        assert Stage.VALID.value == 2
        assert Stage.TEST.value == 3

class TestTrainTestPair:
    def test_print_shapes(self):
        # checking for numpy array
        x = np.array([1,2,3])
        y = np.array([2,3,4])
        stage = Stage.TRAIN
        ttp = FeatureTargetPair(x, y, stage)
        ttp.print_shapes()

        # checking for pandas dataframe
        df = pd.DataFrame({ "x": [1,2,3], "y": [3,4,5] })
        stage = Stage.VALID
        ttp = FeatureTargetPair(df["x"], df["y"], stage)
        ttp.print_shapes()





