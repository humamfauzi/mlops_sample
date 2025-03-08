from enum import Enum, classmethod
from column.abc import Column
from typing import List

# NOTE: the number in enumerate should correspond to column number it will later replaced
class CommodityFlow(Enum, Column):
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
    def primary_id(cls):
        return cls.SHIPMENT_ID

    @classmethod
    def target(cls):
        return cls.SHIPMENT_VALUE

    @classmethod
    def categorical(cls):
        return [
            cls.ORIGIN_STATE,
            cls.ORIGIN_DISTRICT,
            cls.ORIGIN_CFS_AREA,

            cls.DESTINATION_STATE,
            cls.DESTINATION_DISTRICT, 
            cls.DESTINATION_CFS_AREA,

            cls.NAICS,
            cls.QUARTER,
            cls.SCTG,
            cls.MODE,

            cls.EXPORT_COUNTRY,
            cls.HAZMAT,
        ]

    @classmethod
    def numerical(cls):
        return [
            cls.SHIPMENT_WEIGHT, 
            cls.SHIPMENT_DISTANCE_CIRCLE,
            cls.SHIPMENT_DISTANCE_ROUTE,
            cls.WEIGHT_FACTOR,
        ]

    @classmethod
    def feature(cls, current_column):
        """
        list all feature based on what current existing column.
        used when we drop some column from the original dataset
        """
        all = cls.numerical() + cls.categorical()
        return list(set(all) & set(current_column))



