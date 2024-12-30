from abc import ABC, abstractmethod
from train.column import CommodityFlow
import pandas as pd
from copy import copy

class TabularDataCleaner(ABC):
    @abstractmethod
    def clean_data(self) -> pd.DataFrame:
        """
        Clean all data inside the data frame
        regardless the input, it should have output of dataframe
        """
        pass


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
            CommodityFlow.ORIGIN_DISTRICT,
            CommodityFlow.ORIGIN_CFS_AREA,
            CommodityFlow.DESTINATION_DISTRICT,
            CommodityFlow.DESTINATION_CFS_AREA,

            # we use NAICS as goods categarization
            CommodityFlow.SCTG,
            CommodityFlow.QUARTER,

            # we use actual route instead of geodesic distance
            CommodityFlow.SHIPMENT_DISTANCE_CIRCLE,

            # we disable all options
            CommodityFlow.IS_TEMPERATURE_CONTROLLED,
            CommodityFlow.IS_EXPORT,
            CommodityFlow.EXPORT_COUNTRY,
            CommodityFlow.HAZMAT,
            CommodityFlow.WEIGHT_FACTOR,
        ]

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.cleaned_data = df
        (self
            .remove_columns(self.basic_removal())
            .remove_nan_rows()
         )
        return copy(self.cleaned_data)
    
    def remove_columns(self, columns):
        self.cleaned_data.drop(columns, axis=1, inplace=True)
        return self

    def remove_nan_rows(self):
        self.cleaned_data.dropna(inplace=True)
        return self
