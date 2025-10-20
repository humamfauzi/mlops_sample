from abc import ABC, abstractmethod
from train.column import TabularColumn
import pandas as pd
from copy import copy
from typing import Optional
import time

class TabularDataCleaner(ABC):
    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean all data inside the data frame
        regardless the input, it should have output of dataframe
        """
        pass

class Cleaner(TabularDataCleaner):
    '''
    Data cleaning with lazy call. Means that the data cleaning will be done
    after the data is called. This is useful when we want to chain the data cleaning.
    Especially if we want to use it in main level for more flexibility.
    Any change regarding data cleaning can be done in main level instead of changing class in
    data_cleaner.py

    Note: In essence, "lazy call" means delaying the execution of a function or operation until it is explicitly triggered, 
    providing a way to accumulate and manage a series of actions to be performed later as a single unit.
    '''
    def __init__(self, facade):
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.call_container = []
        self.facade = facade

    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        start = time.time()
        self.cleaned_data = df.copy()
        for call in self.call_container:
            call()
        self.reparse_data_type()
        time_ms = int((time.time() - start) * 1000)
        self.write_metadata(time_ms)
        return copy(self.cleaned_data)

    def reparse_data_type(self):
        for col in self.cleaned_data.columns:
            if col in self.column.numerical():
                self.cleaned_data[col] = pd.to_numeric(self.cleaned_data[col], errors='raise')
            if col in self.column.categorical():
                self.cleaned_data[col] = self.cleaned_data[col].astype('category')
        return self

    def write_metadata(self, time_ms):
        if self.facade is None:
            return self
        self.facade.set_data_cleaning_time(time_ms)
        self.facade.set_row_size_after_cleaning(len(self.cleaned_data.index))
        self.facade.set_column_size_after_cleaning(len(self.cleaned_data.columns))
        return self

    def remove_columns(self, columns):
        self.call_container.append(lambda: self.cleaned_data.drop(columns, axis=1, inplace=True))
        return self

    def filter_columns(self, columns):
        def filter_columns():
            self.cleaned_data = self.cleaned_data[columns]
        self.call_container.append(filter_columns)
        return self

    def remove_nan_rows(self):
        self.call_container.append(lambda: self.cleaned_data.dropna(inplace=True))
        return self

    @classmethod
    def parse_instruction(cls, properties: dict, call: list, facade):
        c = cls(facade)
        c.column = TabularColumn.from_string(properties.get("reference"))
        for step in call:
            cols = [c.column.from_enum(col).name for col in step.get("columns", [])]
            if step["type"] == "remove_columns":
                c.remove_columns(cols)
            elif step["type"] == "filter_columns":
                c.filter_columns(cols)
            elif step["type"] == "remove_nan_rows":
                c.remove_nan_rows()
        return c

    def execute(self, input): 
        if input is None:
            raise ValueError("Data Cleaner should not be the first step therefore it should have input")
        data = self.clean_data(input)
        return data