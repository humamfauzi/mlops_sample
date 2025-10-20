from column.cfs2017 import CommodityFlow

class TestDataTransform:
    def test_check_inclusion(self):
        assert CommodityFlow.SHIPMENT_VALUE.name in CommodityFlow.numerical()

    def test_variable_inclusion(self):
        asd = CommodityFlow
        assert CommodityFlow.SHIPMENT_WEIGHT.name in asd.numerical()