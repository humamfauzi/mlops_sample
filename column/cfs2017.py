from enum import Enum
from abc import ABC, abstractmethod
from typing import List

# a join between primary id, target, categorical, and numerical should always be
# a full column
class TabularColumn(ABC):
    # a dataframe should always have primary identifier
    # that unique in all rows
    @classmethod
    @abstractmethod
    def primary_id(cls):
        pass

    # a training dataset should also have a target
    @classmethod
    @abstractmethod
    def target(cls) -> str:
        pass

    # all tabular dataset either a categorical or numerical
    # this methods will call all categorical column
    @classmethod
    @abstractmethod
    def categorical(cls):
        pass

    # all tabular dataset either a categorical or numerical
    # this methods will call all numerical column
    @classmethod
    @abstractmethod
    def numerical(cls):
        pass

    # The reason why you need current column input because
    # there is high chance that you dont use all column because one or other thing
    # there fore we need to know the current input and find intersect
    # between it and all possible feature
    @classmethod
    @abstractmethod
    def feature(cls, current_column: List[Enum]):
        pass

    @classmethod
    def from_string(cls, name: str):
        if name == "commodity_flow":
            return CommodityFlow
        elif name == "sample":
            return SampleEnum
        elif name == "sample_enum_transformer":
            return SampleEnumTransformer
        else:
            raise ValueError(f"Cannot find enum with name {name}")

class SampleEnum(Enum):
    COLUMN_ID = 1
    COLUMN_FEATURE = 2
    COLUMN_FEATURE_DELETED = 3 
    COLUMN_TARGET = 4
    COLUMN_REMOVED = 5

    @classmethod
    def from_enum(cls, e:str):
        for member in cls:
            if member.name == e.upper():
                return member
        raise ValueError(f"Cannot find enum with name {e}")

    @classmethod
    def primary_id(cls):
        return cls.COLUMN_ID.name

    @classmethod
    def target(cls):
        return cls.COLUMN_TARGET.name

    @classmethod
    def categorical(cls):
        return []

    @classmethod
    def numerical(cls):
        return [
            cls.COLUMN_FEATURE.name,
            cls.COLUMN_TARGET.name
        ]

    @classmethod
    def feature(cls, current_column):
        """
        """
        all = cls.numerical() + cls.categorical()
        return list(set(all) & set(current_column))

# NOTE: the number in enumerate should correspond to column number it will later replaced
# all usage should be using its name therefore the column would be full capital
class CommodityFlow(Enum):
    SHIPMENT_ID = 1

    # State idenfitifer using FIPS state code
    ORIGIN_STATE = 2
    ORIGIN_DISTRICT = 3
    # concanation of state and district
    ORIGIN_CFS_AREA = 4
    
    # State idenfitifer using FIPS state code
    DESTINATION_STATE = 5
    DESTINATION_DISTRICT = 6
    # concanation of state and district
    DESTINATION_CFS_AREA = 7

    NAICS = 8 # north american industry classification system
    QUARTER = 9 # Q1, Q2, Q3 or Q4
    SCTG = 10 # standard classification of transported good
    MODE = 11 # Transportation means like truck, ship, airplane etc.

    SHIPMENT_VALUE = 12 # Shipment value measured in dollar
    SHIPMENT_WEIGHT = 13 # Shipment weight in pounds

    # Geodesic straight line from one point to another point; measured in miles
    SHIPMENT_DISTANCE_CIRCLE = 14
    # Actual routing of shipment; measured in miles
    SHIPMENT_DISTANCE_ROUTE = 15 

    # is the shipment have a deliberate temperature control
    IS_TEMPERATURE_CONTROLLED = 16 
    IS_EXPORT = 17 # is the shipment intended for export
    EXPORT_COUNTRY = 18
    HAZMAT = 19
    WEIGHT_FACTOR = 20

    @classmethod
    def from_enum(cls, e:str):
        for member in cls:
            if member.name == e.upper():
                return member
        raise ValueError(f"Cannot find enum with name ")

    @classmethod
    def primary_id(cls):
        return cls.SHIPMENT_ID.name

    @classmethod
    def target(cls):
        return cls.SHIPMENT_VALUE.name

    @classmethod
    def categorical(cls):
        return [
            cls.ORIGIN_STATE.name,
            cls.ORIGIN_DISTRICT.name,
            cls.ORIGIN_CFS_AREA.name,

            cls.DESTINATION_STATE.name,
            cls.DESTINATION_DISTRICT.name, 
            cls.DESTINATION_CFS_AREA.name,

            cls.NAICS.name,
            cls.QUARTER.name,
            cls.SCTG.name,
            cls.MODE.name,

            cls.EXPORT_COUNTRY.name,
            cls.HAZMAT.name,
        ]

    @classmethod
    def numerical(cls):
        return [
            cls.SHIPMENT_WEIGHT.name, 
            cls.SHIPMENT_DISTANCE_CIRCLE.name,
            cls.SHIPMENT_DISTANCE_ROUTE.name,
            cls.WEIGHT_FACTOR.name,
            cls.SHIPMENT_VALUE.name,
        ]

    @classmethod
    def feature(cls, current_column):
        """
        """
        all = cls.numerical() + cls.categorical()
        return list(set(all) & set(current_column))



class SampleEnumTransformer(Enum):
    COLUMN_ID = 1
    COLUMN_CATEGORICAL = 2
    COLUMN_NUMERICAL = 3
    COLUMN_TARGET = 4

    @classmethod
    def categorical(cls):
        return [cls.COLUMN_CATEGORICAL.name]

    @classmethod
    def numerical(cls):
        return [cls.COLUMN_NUMERICAL.name]

    @classmethod
    def feature(cls, current_column):
        alll = cls.numerical() + cls.categorical()
        return list(set(alll) & set(current_column))

    @classmethod
    def target(cls):
        return cls.COLUMN_TARGET.name

    @classmethod
    def from_enum(cls, e:str):
        for member in cls:
            if member.name == e.upper():
                return member
        raise ValueError(f"Cannot find enum with name {e}")