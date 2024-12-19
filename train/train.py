import pandas as pd
from datacleaner import DataCleaner
# from dataio import DataIO


# Load data and provide other classes with read data capability
# all raed centralized here so there would be no duplicate raw data read
class DataManager:
    def __init__(self, path):
        self.path = path
        self.raw_data = None

        # the data itself pretty large, we need to set the default chunk size
        self.chunk_size = 1000
        return

    def set_chunk_size(self, chunk):
        self.chunk_size = chunk
        return self

    def load_dataset(self):
        self.raw_data = pd.read_csv("dataset/cfs_2017.csv", chunksize=self.chunk_size)
        return self

    def get_dataset(self):
        if self.raw_data is None:
            return self.load_dataset().raw_data
        return self.raw_data

    def write_dataframe(self, dataframe):
        pass


class Prelim:
    def __init__(self):
        self.raw_data = None

    def load_dataset(self, chunksize=1000):
        self.raw_data = pd.read_csv("dataset/cfs_2017.csv", nrows=chunksize)
        return self


if __name__ == "__main__":
    pre = Prelim()
    pre.load_dataset()
    print(">>>>>", pre.raw_data.columns)
    dc = DataCleaner(pre.raw_data)
    dc.alter_column()
    print("hello")
    print(dc.df.columns)
    print("hello2")
