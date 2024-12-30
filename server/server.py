import http.server
import socketserver 
import mlflow
import os
import numpy as np
import pickle
import time
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse, parse_qs

from train.column import CommodityFlow
import json

PORT = 5050
# TODO rather than become constant, it should be injected/input to handler class for
# more granular control
ARTIFACT_DIR="server"

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        start_time = time.time()
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)

        if parsed_path.path == '/cfs2017':
            result = self.process_query(query_params)
            self.respond_with_result(result)
        elif parsed_path.path == '/health':
            self.respond_with_check()
        else:
            self.respond_with_not_found()
        print(f"done writing request {(time.time() - start_time):.6f}s")
        return
    
    def process_query(self, query_params):
        self.precheck(query_params)
        transformed = self.transform(query_params)
        cost = self.infer(transformed)
        return {
            "message": "success",
            "data": {
                "cost": cost,
            }
        }

    def respond_with_result(self, result):
        response = json.dumps(result)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
        return

    def respond_with_check(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"OK")
        return

    def respond_with_not_found(self):
        self.send_response(404)

    def precheck(self, raw):
        important = [
            "origin_state",
            "destination_state",
            "naics",
            "mode",
            "shipment_weight",
            "shipment_distance_route",
        ]
        for i in important:
            if i not in raw.keys():
                raise ValueError(f"Require {i} but not provided")
        return self

    def infer(self, input):
        directory_path = f"{ARTIFACT_DIR}/artifacts/cfs_model/model.pkl"
        with open(directory_path, 'rb') as f:
            model = pickle.load(f)
            return model.predict(input)[0]
            
    def transform(self, raw):
        # column order should match the input transformation in train/train.py
        mapp = {
            "origin_state": CommodityFlow.ORIGIN_STATE,
            "destination_state": CommodityFlow.DESTINATION_STATE,
            "naics": CommodityFlow.NAICS,
            "mode": CommodityFlow.MODE,
            "shipment_weight": CommodityFlow.SHIPMENT_WEIGHT,
            "shipment_distance_route": CommodityFlow.SHIPMENT_DISTANCE_ROUTE,
        }
        container = []
        for key, value in raw.items():
            new_value = self.implement_preprocess(mapp[key], value)
            container.append(new_value)
        return np.concat(container, axis=1)

    def implement_preprocess(self, col, original_value):
        # TODO it need loaded once when the handler start running
        # so any request just use loaded pickle; not reading from disk
        directory_path = f"{ARTIFACT_DIR}/artifacts/preprocess/{col}"
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        # if there is no files, it means there are no transformation for that column.
        # so we just return the original value
        for file in files:
            with open(f"{directory_path}/{file}", 'rb') as f:
                encoder = pickle.load(f)
                return self.handle_encoder(encoder, original_value)
        return original_value

    def handle_encoder(self, encoder, value):
        return encoder.transform(np.array(value).reshape(1, -1).astype(np.int64))

# MLFlow class handling loading artifacts for server.
class MlFlowManagement:
    def __init__(self, tracking_url, name):
        self.tracking_url = tracking_url
        self.experiment = name
        self.client = None

    def begin(self):
        mlflow.set_tracking_uri(uri="http://mlflow:5000")
        mlflow.set_experiment("humamtest")
        return self

    def set_client(self):
        client = MlflowClient()
        self.client = client
        return self

    def set_artifact_destination(self, destination):
        self.artifact_destination = destination
        return self

    def find_latest_run(self):
        experiment_id = 1
        runs = self.client.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1)
        if runs:
            latest_run_id = runs[0].info.run_id
            return latest_run_id
        else:
            raise ValueError("failed to find the latest run")

    def find_desired_run_id(self):
        # TODO should find a training result with production tag
        # for now use the latest run 
        return self.find_latest_run()

    def columns(self):
        return [
            CommodityFlow.DESTINATION_STATE,
            CommodityFlow.ORIGIN_STATE,
            CommodityFlow.MODE,
            CommodityFlow.SHIPMENT_DISTANCE_ROUTE,
            CommodityFlow.NAICS,
            CommodityFlow.SHIPMENT_WEIGHT
        ]
    def load_all_artifacts(self):
        run_id = self.find_desired_run_id()
        uri = f"mlflow-artifacts:/1/{run_id}/artifacts"
        all = mlflow.artifacts.list_artifacts(artifact_uri=uri)
        mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=f"{self.artifact_destination}/artifacts")
        return self

TRACKER_PATH = "http://mlflow:5000" # see docker compose for details
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    # The strategy here is that we want to load all required pkl and other artifact once
    # so that the http handler can just read the disk for inference
    mgmt = (MlFlowManagement(TRACKER_PATH, "humamtest")
        .begin()
        .set_client()
        .set_artifact_destination(ARTIFACT_DIR)
        .load_all_artifacts()
    )
    print(f"Serving at port {PORT}")
    # TODO need to handle keyboard interrupt for graceful shutdown
    httpd.serve_forever()
