import pandas as pd
from enum import Enum
from column import CommodityFlow
from sklearn.preprocessing import OneHotEncoding
from sklearn.model_selection import train_test_split
class Part:
    TRAIN = 1
    VALID = 2
    TEST = 3

class TrainPair:
    def __init__(self, X, y, part):
        self.part = part
        self.target = y
        self.train = X

# handling any transoformation data set
# final product should be train, valid, and test dataset
class DataTransform:
    def __init__(self, df, target):
        self.rand = 42
        self.df = df
        self.train = None
        self.valid = None
        self.test = None

        self.train_proportion = .8
        self.valid_proportion = .1
        self.test_proportion = .1
        
    def split(self):
        validtest_proportion = self.valid_propertion + self.test_proportion
        Xtr, ytr, Xte, yte = train_test_split(
            self.train,
            self.target, 
            test_size=validtest_proportion, 
            random_state=self.rand
        )
        self.train = TrainPair(Xtr, ytr, Part.TRAIN)
        Xval, yval, Xtest, ytest= train_test_split(
            Xte, 
            yte, 
            test_size=self.test_proportion,
            random_state=self.rand
        )
        self.valid = TrainPair(Xval, yval, Part.VALID)
        self.test = TrainPair(Xtest, ytest, Part.TEST)
        return self

