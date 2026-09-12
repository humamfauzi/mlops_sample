import json
import pathlib
from urllib.parse import urlencode

from locust import HttpUser, task

# The load test targets one published model. Query keys must match the
# uppercase enum names in that model's input manifest, and values must fall
# inside the range / available_values recorded there at training time.
# Inspect the live contract with:  GET /cfs2017
#
# The target lives in target.json so CI can verify the model is actually
# served before generating traffic against it.
TARGET = json.loads((pathlib.Path(__file__).parent / "target.json").read_text())
MODEL = TARGET["model"]
QUERY_PARAMS = TARGET["query"]


class User(HttpUser):
    @task
    def my_task(self):
        base = f"/cfs2017/{MODEL}/inference"
        full = f"{base}?{urlencode(QUERY_PARAMS)}"
        # max request just three second; consider it fail if above three second
        self.client.get(full, timeout=3)
