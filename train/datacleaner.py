import pandas as pd
from column import CommodityFlow

class DataCleaner:
    def __init__(self, df):
        self.df = df

    def get_df(self):
        # use copy so any alteration by getter does not affect the self.df
        return copy(self.df)
            
    # every cleaner should call alter_column to transfer column name
    def alter_column(self):
        column_length = len(self.df.columns)
        enum_length = len(list(CommodityFlow))
        if column_length != enum_length:
            err_msg = f"Column in dataframe {column_length} is different than enum {enum_elenth}"
            raise ValueError(err_msg)
        col_dict = {self.df.columns[i]: CommodityFlow(i+1) for i in range(column_length)}
        self.df = self.df.rename(columns=col_dict)
        return self

    def remove_columns(self, *args):
        self.df = self.df.drop(columns=args)
        return self

    # remove all rows that contain null
    def remove_null(self, columns):
        self.df = self.df.dropna(subset=columns)
        return self

    # remove all outlier value
    def remove_nonsense_value(self):
        pass



