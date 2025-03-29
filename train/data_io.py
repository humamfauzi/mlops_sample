from abc import ABC, abstractmethod
import os
import pandas as pd
import mlflow
from copy import copy
from typing import Optional
from train.sstruct import Pairs, Stage, FeatureTargetPair
from train.column import TabularColumn
from repositories.mlflow import MLflowRepository
from enum import Enum

class Disk:
    def __init__(self, path, name: str, repository: MLflowRepository = None):
        self.path = path
        self.name = name
        self.loader: Optional[function] = None
        self.saver: Optional[function] = None
        self.repository: Optional[MLflowRepository] = repository

    def load_dataframe_via_csv(self, column: TabularColumn, load_options: dict):
        def loader() -> pd.DataFrame:
            raw_data = pd.read_csv(f"{self.path}/{self.name}.csv", **load_options)
            defined = mlflow.data.from_pandas(raw_data, name=self.name)
            mlflow.log_input(defined)
            raw_data = self._replace_columns(raw_data, column)
            return copy(raw_data)
        self.loader = loader
        return self
        
    def load_data(self):
        if self.loader is None:
            raise ValueError("should define the loading method first")
        return self.loader()

    def save_via_csv(self):
        def saver(data: pd.DataFrame):
            fulldir = f'{self.path}/{self.name}.csv'
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            data.to_csv(fulldir, index=False)
        self.saver = saver
        return self
    
    def save_data(self, data):
        if self.saver is None:
            raise ValueError("should define the saving method first")
        self.saver(data)
    
    def save_pair_via_parquet(self):
        def saver(pairs):
            if not isinstance(pairs, Pairs):
                raise TypeError("Input data must be of type Pairs")
            base = f"{self.path}/{self.name}"
            (self._save_parquet_data(pairs.train.X, f"{base}/train/feature.parquet")
                ._save_parquet_data(pairs.train.y.to_frame(), f"{base}/train/target.parquet")
                ._save_parquet_data(pairs.valid.X, f"{base}/valid/feature.parquet")
                ._save_parquet_data(pairs.valid.y.to_frame(), f"{base}/valid/target.parquet")
                ._save_parquet_data(pairs.test.X, f"{base}/test/feature.parquet")
                ._save_parquet_data(pairs.test.y.to_frame(), f"{base}/test/target.parquet"))
        self.saver = saver
        return self

    def _save_parquet_data(self, df: pd.DataFrame, path: str):
        dir = "/".join(path.split("/")[:-1])
        if not os.path.exists(dir):
            os.makedirs(dir)
        df.to_parquet(path, index=False)
        return self

    def load_pair_via_parquet(self):
        def loader() -> pd.DataFrame:
            base = f"{self.path}/{self.name}"
            xtr = pd.read_parquet(f"{base}/train/feature.parquet")
            ytr = pd.read_parquet(f"{base}/train/target.parquet")
            ftrain = FeatureTargetPair(xtr, ytr, Stage.TRAIN)

            xval = pd.read_parquet(f"{base}/valid/feature.parquet")
            yval = pd.read_parquet(f"{base}/valid/target.parquet")
            fvalid = FeatureTargetPair(xval, yval, Stage.VALID)

            xte = pd.read_parquet(f"{base}/test/feature.parquet")
            yte = pd.read_parquet(f"{base}/test/target.parquet")
            ftest = FeatureTargetPair(xte, yte, Stage.TEST)
            defined = mlflow.data.from_pandas(xtr, name=self.path)
            mlflow.log_input(defined)
            return Pairs(ftrain, fvalid, ftest)
        self.loader = loader
        return self

    @staticmethod
    def _replace_columns(data: pd.DataFrame, enum: TabularColumn):
        Disk._check_length(data, enum)
        # it has minus one because index in python began with 0
        replace_map = {data.columns[e.value-1]:e for e in enum}
        data.rename(columns=replace_map, inplace=True)
        return data

    @staticmethod
    def _check_length(data: pd.DataFrame, enum: TabularColumn):
        if data is None:
            raise ValueError("raw data need to be loaded first")
        if len(enum) != len(data.columns):
            raise ValueError(f"Cannot replace columns: enum {len(enum)} df {len(data.columns)}")
        return data