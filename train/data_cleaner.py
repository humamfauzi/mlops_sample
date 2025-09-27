from abc import ABC, abstractmethod
from train.column import TabularColumn
import pandas as pd
from copy import copy
from typing import Optional

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
    def __init__(self):
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.call_container = []
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.cleaned_data = df.copy()
        for call in self.call_container:
            call()
        return copy(self.cleaned_data)

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
    def parse_instruction(cls, properties: dict, call: list):
        c = cls()
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