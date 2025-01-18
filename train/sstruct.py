from enum import Enum
import numpy as np
import pandas as pd

class Stage(Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3

class FeatureTargetPair:
    def __init__(
        self, 
        X: pd.DataFrame, 
        y: pd.DataFrame, 
        stage: Stage
    ) -> None:
        self.X = X
        self.y = y
        self.stage = stage

    def print_shapes(self):
        print("stages", self.stage)
        print("X", self.X.shape)
        print("y", self.y.shape)

    def x_array(self) -> np.ndarray:
        return np.array(self.X)

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
