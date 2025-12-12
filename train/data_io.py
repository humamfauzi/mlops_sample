from abc import ABC, abstractmethod
import os
import pandas as pd
from copy import copy
from typing import Optional, List
from train.sstruct import Pairs, Stage, FeatureTargetPair
import numpy as np
from train.column import TabularColumn
from enum import Enum
import time

class Disk:
    def __init__(self, facade, path, name: str, loader=None, saver=None, column=None):
        self.facade = facade
        self.path = path
        self.name = name
        # TODO change loader and saver to be an array of functions
        self.loader: Optional[function] = loader
        self.saver: Optional[function] = saver
        self.column: Optional[TabularColumn] = column

    def load_dataframe_via_csv(self, column: TabularColumn, load_options: dict):
        def loader() -> pd.DataFrame:
            start = time.time()
            raw_data = pd.read_csv(f"{self.path}/{self.name}.csv", **load_options)
            time_ms = int((time.time() - start) * 1000)
            raw_data = self._replace_columns(raw_data, column)
            self.write_metadata(raw_data, time_ms)
            return copy(raw_data)
        self.loader = loader
        return self

    def load_random_rows_via_csv(self, column: TabularColumn, load_options: dict, n_rows: int, random_state: int = None):
        """
        Loads a DataFrame from CSV, picking n_rows randomly from the entire file (not just the first n rows).
        :param column: TabularColumn for column replacement
        :param load_options: dict of options for pd.read_csv (excluding nrows and skiprows)
        :param n_rows: number of random rows to sample
        :param random_state: random seed for reproducibility
        """
        def loader() -> pd.DataFrame:
            csv_path = f"{self.path}/{self.name}.csv"
            # First, count the number of lines (excluding header)
            with open(csv_path, 'r') as f:
                total_lines = sum(1 for _ in f) - 1
            if n_rows > total_lines:
                raise ValueError(f"Requested n_rows={n_rows} but file only has {total_lines} rows.")
            skiprows = self.generate_skiprows_function(total_lines, n_rows, random_state)
            raw_data = pd.read_csv(csv_path, skiprows=skiprows, **load_options)
            raw_data = self._replace_columns(raw_data, column)
            return copy(raw_data)
        self.loader = loader
        return self

    def generate_skiprows_function(self, total_lines: int, n_rows: int, random_state: int = None):
        if n_rows > total_lines:
            raise ValueError(f"Requested n_rows={n_rows} but file only has {total_lines} rows.")
        rng = np.random.default_rng(random_state)
        chosen = set(rng.choice(total_lines, n_rows, replace=False) + 1)
        skip = set(range(1, total_lines + 1)) - chosen
        def skiprows(i):
            return i in skip and i != 0
        return skiprows


    def write_metadata(self, raw_data, time_ms):
        if self.facade is None:
            return self
        self.facade.set_data_loading_time(time_ms)
        self.facade.set_row_size(len(raw_data.index))
        self.facade.set_column_size(len(raw_data.columns))
        self.facade.set_dataset_name(self.name)
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
            pairs.train.str_columns()
            pairs.valid.str_columns()
            pairs.test.str_columns()
            (self._save_parquet_data(pairs.train.X, f"{base}/train/feature.parquet")
                ._save_parquet_data(pairs.train.y, f"{base}/train/target.parquet")
                ._save_parquet_data(pairs.valid.X, f"{base}/valid/feature.parquet")
                ._save_parquet_data(pairs.valid.y, f"{base}/valid/target.parquet")
                ._save_parquet_data(pairs.test.X, f"{base}/test/feature.parquet")
                ._save_parquet_data(pairs.test.y, f"{base}/test/target.parquet"))
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
            return Pairs(ftrain, fvalid, ftest)
        self.loader = loader
        return self

    @staticmethod
    def _replace_columns(data: pd.DataFrame, enum: TabularColumn):
        Disk._check_length(data, enum)
        # it has minus one because index in python began with 0
        replace_map = {data.columns[e.value-1]:e.name for e in enum}
        data.rename(columns=replace_map, inplace=True)
        return data

    @staticmethod
    def _check_length(data: pd.DataFrame, enum: TabularColumn):
        if data is None:
            raise ValueError("raw data need to be loaded first")
        if len(enum) != len(data.columns):
            raise ValueError(f"Cannot replace columns: enum {len(enum)} df {len(data.columns)}")
        return data

    @classmethod
    def parse_instruction(cls, properties: dict, call: List[dict], facade):
        # ignore the properties
        c = cls(facade, properties.get("path"), properties.get("file"))
        c.column = TabularColumn.from_string(properties.get("reference"))
        for step in call:
            if step["type"] == "load":
                c.load_dataframe_via_csv(c.column, {"nrows": step.get("n_rows", None)})
        return c
        
    def execute(self, input_data):
        if input_data is not None:
            raise ValueError("Data IO should be the first step therefore it should not have any input")
        data = self.load_data()
        return data