from abc import ABC, abstractmethod
from enum import Enum
import itertools
import os
import pandas as pd
from copy import copy
from typing import Optional
from train.sstruct import Pairs, Stage, FeatureTargetPair

class TabularDataLoader(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Load all data based on the information given in initization
        the output should be a pandas data frame
        """
        pass

    @abstractmethod
    def save_data(self, df: pd.DataFrame, output: str):
        return self

    @abstractmethod
    def save_pairs(self, directory: str, pairs: Pairs):
        pass

    @abstractmethod
    def load_pairs(self, directory: str) -> Pairs:
        pass

# load data from disk or other source to programs
# alter the column to enumerated column name
# it should transform whatever data in disk to a pandas dataframe
# because it has abc of TabularDataLoader
class Disk(TabularDataLoader):
    def __init__(self, path: str, enum: Enum, chunk=None):
        self.path = path
        self.raw_data: Optional[pd.DataFrame] = None
        self.chunk_size = chunk
        self.enum = enum

    def load_data(self):
        kwargs = {}
        if self.chunk_size is not None:
            kwargs['nrows'] = self.chunk_size
        self.raw_data = pd.read_csv(self.path, **kwargs)
        self.replace_columns()
        return copy(self.raw_data)

    def replace_columns(self):
        if self.raw_data is None:
            raise ValueError("raw data need to be loaded first")
        self.check_length()
        # it has minus one because index in python began with 0
        replace_map = {self.raw_data.columns[e.value-1]:e for e in self.enum}
        self.raw_data.rename(columns=replace_map, inplace=True)
        return self

    def check_length(self):
        if self.raw_data is None:
            raise ValueError("raw data need to be loaded first")
        if len(self.enum) != len(self.raw_data.columns):
            raise ValueError(f"Cannot replace columns: enum {len(self.enum)} df {len(self.raw_data.columns)}")
        return self

    def load_pairs(self, directory: str) -> Pairs:
        xtr = pd.read_parquet(f"dataset/{directory}/train/feature.parquet")
        ytr = pd.read_parquet(f"dataset/{directory}/train/target.parquet")
        ftrain = FeatureTargetPair(xtr, ytr, Stage.TRAIN)

        xval = pd.read_parquet(f"dataset/{directory}/valid/feature.parquet")
        yval = pd.read_parquet(f"dataset/{directory}/valid/target.parquet")
        fvalid = FeatureTargetPair(xval, yval, Stage.VALID)

        xte = pd.read_parquet(f"dataset/{directory}/test/feature.parquet")
        yte = pd.read_parquet(f"dataset/{directory}/test/target.parquet")
        ftest = FeatureTargetPair(xte, yte, Stage.TEST)
        return Pairs(ftrain, fvalid, ftest)

    def save_pairs(self, directory: str, pairs: Pairs):
        self.save_data(pairs.train.X, f"{directory}/train/feature.parquet")
        self.save_data(pairs.train.y.to_frame(), f"{directory}/train/target.parquet")
        self.save_data(pairs.valid.X, f"{directory}/valid/feature.parquet")
        self.save_data(pairs.valid.y.to_frame(), f"{directory}/valid/target.parquet")
        self.save_data(pairs.test.X, f"{directory}/test/feature.parquet")
        self.save_data(pairs.test.y.to_frame(), f"{directory}/test/target.parquet")

    def save_data(self, df: pd.DataFrame, output: str):
        path = f'dataset/{output}'
        dir = "/".join(path.split("/")[:-1])
        if not os.path.exists(dir):
           os.makedirs(dir)
        df.to_parquet(path, index=False)
        return self
