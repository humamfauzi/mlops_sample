from enum import Enum
import numpy as np
import pandas as pd

class Stage(Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3

    @classmethod
    def _map(cls):
        return {
            "train": cls.TRAIN,
            "test": cls.TEST,
            "valid": cls.VALID,
        }

    @staticmethod
    def from_str(name: str):
        return Stage._map()[name]

    @classmethod
    def from_enum(cls, name: str):
        return {
            "train": cls.TRAIN,
            "test": cls.TEST,
            "valid": cls.VALID,
        }[name]

class FeatureTargetPair:
    def __init__(
        self, 
        X: pd.DataFrame, 
        y: pd.DataFrame, 
        stage: Stage,
        y_raw: pd.DataFrame = None,
    ) -> None:
        self.X = X
        self.y = y
        self.stage = stage
        # The target *before* any transformation. Transformations replace `y`
        # (log, standardise), so anything that needs the target in its original
        # units -- sample weights, calibration checks -- has to read this.
        # Defaults to `y` for callers that build a pair from already-final data.
        if y_raw is None:
            y_raw = y.copy() if hasattr(y, "copy") else y
        self.y_raw = y_raw

    def str_columns(self):
        self.X.columns = [col.name if isinstance(col, Enum) else col for col in self.X.columns]
        self.y.columns = [col.name if isinstance(col, Enum) else col for col in self.y.columns]
        return 

    def print_shapes(self):
        print("stages", self.stage)
        print("X", self.X.shape)
        print("y", self.y.shape)

    def x_array(self) -> np.ndarray:
        return np.array(self.X)
    
    def y_array(self) -> np.ndarray:
        """
        Most target in sklearn only accept array with single dimension
        therefore we need to offer an options to lower single dimension
        """
        return np.array(self.y).reshape(-1)

    def y_raw_array(self) -> np.ndarray:
        """The target in its original units, before any transformation."""
        return np.array(self.y_raw).reshape(-1)

class Pairs:
    def __init__(self, train, valid, test):
        self.train: FeatureTargetPair = train
        self.valid: FeatureTargetPair = valid
        self.test: FeatureTargetPair = test

    def get_train_pair(self):
        return self.train 

    def get_valid_pair(self):
        return self.valid

    def get_test_pair(self):
        return self.test
