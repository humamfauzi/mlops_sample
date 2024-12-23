from enum import Enum
import numpy as np

class Stage(Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3

class TrainTestPair:
    def __init__(self, X, y, stage):
        self.X = X
        self.y = y
        self.stage = stage

    def print_shapes(self):
        print("stages", self.stage)
        print("X", self.X.shape)
        print("y", self.y.shape)

    def x_array(self):
        return np.array(self.X)

class Pairs:
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test

    def get_train_pair(self):
        return self.train 

    def get_valid_pair(self):
        return self.valid

    def get_test_pair(self):
        return self.test
