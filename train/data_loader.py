from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd
from copy import copy

class TabularDataLoader(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Load all data based on the information given in initization
        the output should be a pandas data frame
        """
        pass


# load data from disk or other source to programs
# alter the column to enumerated column name
# it should transform whatever data in disk to a pandas dataframe
# because it has abc of TabularDataLoader
class Disk(TabularDataLoader):
    def __init__(self, path: str, enum: Enum, chunk=1000):
        self.path = path
        self.raw_data = None
        self.chunk_size = chunk
        self.enum = enum

    def load_data(self):
        self.raw_data = pd.read_csv(self.path, nrows=self.chunk_size)
        self.replace_columns()
        return copy(self.raw_data)

    def replace_columns(self):
        self.check_length()
        # it has minus one because index in python began with 0
        replace_map = {self.raw_data.columns[e.value-1]:e for e in self.enum}
        self.raw_data.rename(columns=replace_map, inplace=True)
        return self

    def check_length(self):
        if len(self.enum) != len(self.raw_data.columns):
            raise ValueError(f"Cannot replace columns: enum {len(self.enum)} df {len(self.raw_data.columns)}")
        return self
