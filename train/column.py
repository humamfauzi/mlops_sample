from enum import Enum

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
    

