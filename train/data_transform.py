import pandas as pd
import numpy as np
import os 
import pickle
import mlflow

from abc import ABC, abstractmethod
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from column.cfs2017 import CommodityFlow
from column.abc import TabularColumn

from train.sstruct import Pairs, Stage, FeatureTargetPair

class TabularDataTransform(ABC):
    @abstractmethod
    def transform_data(self, df: pd.DataFrame) -> Pairs:
        pass

    @abstractmethod
    def set_run_name(self, name: str):
        pass

class DataTransform(TabularDataTransform):
    def __init__(self, enum: TabularColumn):
        self.ohe = {}
        self.mm = {}
        self.transformed_data = None

        self.run_name = None

        self.train_pair = None
        self.valid_pair = None
        self.test_pair = None
        self.enum = enum

    def transform_data(self, df) -> Pairs:
        self.transformed_data = df
        (self
            # all data should be splitted first here
            # so all encoding and transformation is based on training data
            # to prevent learning leakage. All transformation happen between
            # splitting and reapplied
            .split_stage()
            
            # each enum should provide which column is a category
            # so this function only call that function and match it
            # with the repsective column

            # dataframe should already use enum based column
            # see data loader for details
            .one_hot_encoding()

            # each enum should provide which column is a numeric
            # so this function only call that function and match it
            # with the repsective column
            # dataframe should already use enum based column
            # see data loader for details
            .minmax()

            # all those transformation now applied to both
            # valid and test data
            .reapply()
            
            # before we proceed, we need to check the shape
            # this is to ensure that rows does not get modified
            # and check column expansion due to one hot encoding
            # all train, valid, and test should have same column count
            .shape_check()

            # we need to save the tranformation because all future incoming
            # inference use the same data structure as raw data
            # in order our model to perform, it need to be transformed first.
            # we save the transformation to a pickle; saved in mlflow instance
            # we will load those instance later when we serve the model as inference
            .save_transformation()
         )

        return self.get_pairs()

    # while the transformation is not a training, we still need to collect
    # artifacts for transformation therefore we need to know the run name
    def set_run_name(self, name: str):
        self.run_name = name
        return self
    
    # pairs always return three part that later accessed by train
    # the type is not particularly important as long as it has
    # feature, target, and stage (train, valid, or test)
    def get_pairs(self) -> Pairs:
        return Pairs(self.train_pair, self.valid_pair, self.test_pair)

    def print_shapes(self):
        if self.train_pair is None:
            raise ValueError("train pair should exist")
        if self.test_pair is None:
            raise ValueError("train pair should exist")
        if self.valid_pair is None:
            raise ValueError("train pair should exist")
        ddict = {
            Stage.TRAIN.name: {
                "X": self.train_pair.X.shape,
                "y": self.train_pair.y.shape,
            },
            Stage.VALID.name: {
                "X": self.valid_pair.X.shape,
                "y": self.valid_pair.y.shape,
            },
            Stage.TEST.name: {
                "X": self.test_pair.X.shape,
                "y": self.test_pair.y.shape,
            } 
        }
        print(pd.DataFrame(ddict))
        return self

    # Currently we dont have insight about the best practice for saving
    # artifacts. Based on our knowledge, we need to store it based on
    # the column. Some might column require more transformation than others
    # current available transformation are one hot encoding and min max scale
    # stored as a map. If we have other transformation, we will add more key.
    # while it not scalable, it is the best we have.
    def save_transformation(self):
        dir = "artifacts/preprocess"
        for key, value in self.ohe.items():
            # TODO should find another way to represent that something stored as ohe
            # or other transformation methods; working with magic string is risky
            self.save_preprocessing_as_object(dir, "ohe", key, value)
        for key, value in self.mm.items():
            self.save_preprocessing_as_object(dir, "mm", key, value)
        # this function transfer all file under dir to mlflow artifact folder
        # called preprocess. Better than upload individual file
        # for testing, we only test the saving folder part
        # TODO maybe we can use mock testing for all mlflow methods
        mlflow.log_artifacts(dir, artifact_path="preprocess")
        return self

    def save_preprocessing_as_object(self, dir, prefix, key, obj):
        os.makedirs(f'{dir}/{key}', exist_ok=True)
        path = f'{dir}/{key}/{prefix}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        return path

    def shape_check(self):
        if self.train_pair is None:
            raise ValueError("train pair should exist")
        if self.test_pair is None:
            raise ValueError("train pair should exist")
        if self.valid_pair is None:
            raise ValueError("train pair should exist")
        train_column_num = self.train_pair.X.shape[1]
        valid_column_num = self.valid_pair.X.shape[1]
        test_column_num = self.test_pair.X.shape[1]
        if train_column_num != valid_column_num:
            raise ValueError("number of column in train and valid should be the same")
        if train_column_num != test_column_num:
            raise ValueError("number of column in train and test should be the same")
        if test_column_num != valid_column_num:
            raise ValueError("number of column in test and valid should be the same")
        return self


    def split_stage(self):
        if self.transformed_data is None:
            raise ValueError("transformed_data should null")

        train_cols = self.enum.feature(self.transformed_data.columns)
        train = self.transformed_data[train_cols]
        test = self.transformed_data[self.enum.target()]
        Xtr, Xt, ytr, yt = train_test_split(
                train,
                test,
                test_size=0.2,
                random_state=42,
        )
        Xv, Xte, yv, yte = train_test_split(
                Xt,
                yt,
                test_size=0.5,
                random_state=42,
        )
        self.train_pair = FeatureTargetPair(Xtr, ytr, Stage.TRAIN)
        self.valid_pair = FeatureTargetPair(Xv, yv, Stage.VALID)
        self.test_pair = FeatureTargetPair(Xte, yte, Stage.TEST)
        return self

    def one_hot_encoding(self):
        for cat in self._catcol(self.train_pair):
            # it is very likely that we dont encode every possible enums
            # because we only use partial data
            self.ohe[cat] = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
            arrayed = self.train_pair.X[[cat]]
            self.ohe[cat].fit(arrayed)
            self._transform_ohe(self.train_pair, cat)
        return self

    def _transform_ohe(self, pair, cat):
        arrayed = pair.X[[cat]]
        transformed = self.ohe[cat].transform(arrayed)
        encoded_columns = self.ohe[cat].get_feature_names_out([cat])
        new_columns = pd.DataFrame(transformed, columns=encoded_columns)
        pair.X.drop(cat, axis=1, inplace=True)
        arrayed = np.concat([np.array(pair.X), new_columns], axis=1)
        all_columns = list(pair.X.columns) + list(new_columns.columns)
        pair.X = pd.DataFrame(arrayed, columns=all_columns)
        return self

    def minmax(self):
        for num in self._numcol(self.train_pair):
            self.mm[num] = MinMaxScaler()
            arrayed = self.train_pair.X[[num]]
            self.mm[num].fit(arrayed)
            self._transform_mm(self.train_pair, num)
        return self

    def _transform_mm(self, pair, num):
        arrayed = pair.X[[num]]
        transformed = self.mm[num].transform(arrayed)
        pair.X[num] = transformed

    # some column might get dropped in data cleaning
    # there fore we need to find intersect between
    # current column and the categorical column
    def _catcol(self, pair):
        return list(set(
            pair.X.columns) & 
            set(self.enum.categorical())
        )
    # same as above but for numerical columns
    def _numcol(self, pair):
        return list(set(
            pair.X.columns) & 
            set(self.enum.numerical())
        )

    def reapply(self):
        for pair in [self.valid_pair, self.test_pair]:
            # we need to validate that the column exist in the dataframe
            # and the column is being transformed
            # putting column that not transformed would raise KeyError
            for cat in list(set(self._catcol(pair)) & set(self.ohe.keys())):
                self._transform_ohe(pair, cat)
            for num in list(set(self._numcol(pair)) & set(self.mm.keys())):
                self._transform_mm(pair, num)
        return self
