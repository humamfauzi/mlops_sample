from locust import HttpUser, task
from urllib.parse import urlencode

class User(HttpUser):
    @task
    def my_task(self):
        base = "/cfs2017"
        query_params = {
                "naics": 1,
                "origin_state": 1,
                "destination_state": 2,
                "mode": 2,
                "shipment_weight": 200,
                "shipment_distance_route":12,
        }
        full = f"{base}?{urlencode(query_params)}"
        self.client.get(full)
