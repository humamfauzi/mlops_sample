from abc import ABC, abstractmethod
import os
import pandas as pd
import mlflow
from copy import copy
from typing import Optional
from train.sstruct import Pairs, Stage, FeatureTargetPair
from train.column import TabularColumn


class TabularDataFrameLoader(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

class TabularDataFrameSaver(ABC):
    @abstractmethod
    def save_data(self, df: pd.DataFrame, output: str):
        return self

class TabularPairsLoader(ABC):
    @abstractmethod
    def load_pairs(self, directory: str) -> Pairs:
        pass

class TabularPairsSaver(ABC):
    @abstractmethod
    def save_pairs(self, directory: str, pairs: Pairs):
        pass

class RandomDiskLoad(TabularDataFrameLoader):
    def __init__(self, enum: TabularColumn, desired_length: int):
        self.enum = enum
        self.raw_data = None
        self.desired_length = desired_length

    def load_data(self):
        pass

class ChunkDiskLoad(TabularDataFrameLoader):
    def __init__(self, enum: TabularColumn, chunk: int):
        self.enum = enum
        self.raw_data = None
        self.chunk = chunk

    def _replace_columns(self, data: pd.DataFrame):
        self._check_length(data)
        # it has minus one because index in python began with 0
        replace_map = {data.columns[e.value-1]:e.name for e in self.enum}
        data.rename(columns=replace_map, inplace=True)
        return data

    def _check_length(self, data: pd.DataFrame):
        if data is None:
            raise ValueError("raw data need to be loaded first")
        if len(self.enum) != len(data.columns):
            raise ValueError(f"Cannot replace columns: enum {len(self.enum)} df {len(self.raw_data.columns)}")
        return self

    def load_data(self):
        kwargs = {}
        if self.chunk_size is not None:
            kwargs['nrows'] = self.chunk_size
        self.raw_data = pd.read_csv(self.path, **kwargs)
        self.raw_data = self._replace_columns(self.raw_data)
        defined = mlflow.data.from_pandas(self.raw_data, name="cfs2017")
        mlflow.log_input(defined)
        return copy(self.raw_data)

class AllDiskSave(TabularDataFrameSaver):
    def __init__(self, path: str, enum: TabularColumn):
        self.path = path
        self.enum = enum

    def save_data(self, df: pd.DataFrame, output: str):
        path = f'dataset/{output}'
        dir = "/".join(path.split("/")[:-1])
        if not os.path.exists(dir):
            os.makedirs(dir)
        df.to_csv(path, index=False)
        return self

class LoadParquetPairsFromDisk(TabularPairsLoader):
    def __init__(self, directory: str):
        self.directory = directory

    def load_pairs(self) -> Pairs:
        xtr = pd.read_parquet(f"dataset/{self.directory}/train/feature.parquet")
        ytr = pd.read_parquet(f"dataset/{self.directory}/train/target.parquet")
        ftrain = FeatureTargetPair(xtr, ytr, Stage.TRAIN)

        xval = pd.read_parquet(f"dataset/{self.directory}/valid/feature.parquet")
        yval = pd.read_parquet(f"dataset/{self.directory}/valid/target.parquet")
        fvalid = FeatureTargetPair(xval, yval, Stage.VALID)

        xte = pd.read_parquet(f"dataset/{self.directory}/test/feature.parquet")
        yte = pd.read_parquet(f"dataset/{self.directory}/test/target.parquet")
        ftest = FeatureTargetPair(xte, yte, Stage.TEST)
        defined = mlflow.data.from_pandas(xtr, name=self.directory)
        mlflow.log_input(defined)
        return Pairs(ftrain, fvalid, ftest)

class SavePairsToDiskAsParquet(TabularPairsSaver):
    def __init__(self, directory: str):
        self.directory = directory
        return 

    def save_pairs(self, pairs: Pairs):
        (self.save_data(pairs.train.X, f"{self.directory}/train/feature.parquet")
            .save_data(pairs.train.y.to_frame(), f"{self.directory}/train/target.parquet")
            .save_data(pairs.valid.X, f"{self.directory}/valid/feature.parquet")
            .save_data(pairs.valid.y.to_frame(), f"{self.directory}/valid/target.parquet")
            .save_data(pairs.test.X, f"{self.directory}/test/feature.parquet")
            .save_data(pairs.test.y.to_frame(), f"{self.directory}/test/target.parquet"))

    def _save_data(self, df: pd.DataFrame, output: str):
        path = f'dataset/{output}'
        dir = "/".join(path.split("/")[:-1])
        if not os.path.exists(dir):
            os.makedirs(dir)
        df.to_parquet(path, index=False)
        return self

# load data from disk or other source to programsj
# alter the column to enumerated column name
# it should transform whatever data in disk to a pandas dataframe
# because it has abc of TabularDataLoader

class Disk(TabularDataLoader):
    def __init__(self, path: str, enum: TabularColumn, chunk=None):
        self.path = path
        self.raw_data: Optional[pd.DataFrame] = None
        self.chunk_size = chunk
        self.enum = enum

    def set_experiment(self, name: str):
        self.experiment_name = name
        return self

    def set_tracking_path(self, path: str):
        self.tracking_path = path
        return self

    def describe_all_numerical_data(self):
        '''
        Provides a statistical summary of the raw data.
        This method returns a statistical summary of the raw data, including
        measures such as mean, standard deviation, min, and max values for each
        column in the dataset. The raw data must be loaded before calling this method.
        Raises:
            ValueError: If the raw data has not been loaded.
        Returns:
            pandas.DataFrame.describe: A DataFrame containing the statistical summary of the raw data.
        '''
        chunk = 100_000
        average = {col: 0 for col in self.enum.numerical()}
        count = {col: 0 for col in self.enum.numerical()}
        stddev = {col: 0 for col in self.enum.numerical()}
        maxx = {col: float('-inf') for col in self.enum.numerical()}
        minn = {col: float('inf') for col in self.enum.numerical()}

        for chunk in pd.read_csv(self.path, chunksize=chunk):
            cchunk = self.replace_columns(chunk)
            count = {col: count[col] + cchunk[col].count() for col in self.enum.numerical()}
            ssum = {col: average[col] * count[col] + cchunk[col].sum() for col in self.enum.numerical()}
            average = {col: (average[col] * count[col] + cchunk[col].mean() * cchunk[col].count()) / (count[col] + cchunk[col].count()) for col in self.enum.numerical()}
            stddev = {col: stddev[col] + ((cchunk[col] - average[col]) ** 2).sum() for col in self.enum.numerical()}
            stddev = {col: (stddev[col] / count[col]) ** 0.5 for col in self.enum.numerical()}
            minn = {col: min(minn[col], cchunk[col].min()) for col in self.enum.numerical()}
            maxx = {col: max(maxx[col], cchunk[col].max()) for col in self.enum.numerical()}
        distt = {   
            "mean": average,
            "count": count,
            "stddev": stddev,
            "sum": ssum,
            "min": minn,
            "max": maxx
        }
        final = {}
        for col in self.enum.numerical():
            final[col] = {
                "mean": distt["mean"][col],
                "count": distt["count"][col],
                "stddev": distt["stddev"][col],
                "sum": distt["sum"][col],
                "min": distt["min"][col],
                "max": distt["max"][col]
            }
        return final
    
    def describe_all_categorical_data(self):
        chunk = 100_000
        unique = {col: set() for col in self.enum.categorical()}
        count = {col: 0 for col in self.enum.categorical()}
        freq = {col: {} for col in self.enum.categorical()}


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
        defined = mlflow.data.from_pandas(xtr, name=directory)
        mlflow.log_input(defined)
        return Pairs(ftrain, fvalid, ftest)

class Disk:
    def __init__(self, path, name: str):
        self.path = path
        self.name = name
        self.loader: Optional[function] = None
        self.saver: Optional[function] = None


    def load_dataframe_via_csv(self, column: TabularColumn, load_options: dict):
        def loader() -> pd.DataFrame:
            raw_data = pd.read_csv(f"{self.path}/{self.name}.csv", **load_options)
            raw_data = self._replace_columns(raw_data, column)
            defined = mlflow.data.from_pandas(raw_data, name=self.name)
            mlflow.log_input(defined)
            return copy(self.raw_data)
        self.loader = loader
        return self
        
    def load_data(self):
        if self.loader is None:
            raise ValueError("should define the loading method first")
        return self.loader()

    def save_via_csv(self, name: str):
        def saver(data: pd.DataFrame):
            path = f'dataset/{name}'
            if not os.path.exists(path):
                os.makedirs(path)
            data.to_csv(path, index=False)
        self.saver = saver
        return self
    
    def save_pair_via_parquet(self):
        def saver(pairs: Pairs):
            base = f"{self.path}/{self.name}"
            (self.save_parquet_data(pairs.train.X, f"{base}/train/feature.parquet")
                .save_parquet_data(pairs.train.y.to_frame(), f"{base}/train/target.parquet")
                .save_parquet_data(pairs.valid.X, f"{base}/valid/feature.parquet")
                .save_parquet_data(pairs.valid.y.to_frame(), f"{base}/valid/target.parquet")
                .save_parquet_data(pairs.test.X, f"{base}/test/feature.parquet")
                .save_parquet_data(pairs.test.y.to_frame(), f"{base}/test/target.parquet"))
        self.saver = saver
        return self

    def _save_parquet_data(self, df: pd.DataFrame, output: str):
        path = f'dataset/{output}'
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
    def replace_columns(data: pd.DataFrame, enum: TabularColumn):
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