from abc import ABC, abstractmethod
from column.cfs2017 import CommodityFlow
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

class DataFrameLazyCall(TabularDataCleaner):
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
        print(self.cleaned_data)
        for call in self.call_container:
            call()
        return copy(self.cleaned_data)

    def remove_columns(self, columns):
        self.call_container.append(lambda: self.cleaned_data.drop(columns, axis=1, inplace=True))
        return self

    def filter_columns(self, columns):
        def filter_columns():
            print("filtering columns", columns, self.cleaned_data.columns)
            self.cleaned_data = self.cleaned_data[columns]
        self.call_container.append(filter_columns)
        return self

    def remove_nan_rows(self):
        self.call_container.append(lambda: self.cleaned_data.dropna(inplace=True))
        return self


class DataFrame(TabularDataCleaner):
    def __init__(self):
        self.cleaned_data: pd.DataFrame = None

    # TODO it should have separate class since we want all dataframe can handle all enums
    # yet we dont know how to structurize it, therefore it stays for now
    # potential canidates is to let the enums decide what column it want to remove
    # different scenario have different methods
    def basic_removal(self):
        return [
            # using states to limit column number
            CommodityFlow.ORIGIN_DISTRICT.name,
            CommodityFlow.ORIGIN_CFS_AREA.name,
            CommodityFlow.DESTINATION_DISTRICT.name,
            CommodityFlow.DESTINATION_CFS_AREA.name,

            # we use NAICS as goods categarization
            CommodityFlow.SCTG.name,
            CommodityFlow.QUARTER.name,

            # we use actual route instead of geodesic distance
            CommodityFlow.SHIPMENT_DISTANCE_CIRCLE.name,

            # we disable all options
            CommodityFlow.IS_TEMPERATURE_CONTROLLED.name,
            CommodityFlow.IS_EXPORT.name,
            CommodityFlow.EXPORT_COUNTRY.name,
            CommodityFlow.HAZMAT.name,
            CommodityFlow.WEIGHT_FACTOR.name,
        ]

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.cleaned_data = df
        (self
            .remove_columns(self.basic_removal())
            .remove_nan_rows())
        return copy(self.cleaned_data)
    
    def remove_columns(self, columns):
        self.cleaned_data.drop(columns, axis=1, inplace=True)
        return self

    def remove_nan_rows(self):
        self.cleaned_data.dropna(inplace=True)
        return self

    def filter_columns(self, columns):
        self.cleaned_data = self.cleaned_data[columns]
        return self
