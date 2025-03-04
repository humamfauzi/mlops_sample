import async_timeout
import asyncio
import os
import pickle
from mlflow.tracking import MlflowClient
import numpy as np
from types import Dict
import mlflow
from fastapi import FastAPI, Request, Query
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from train.column import CommodityFlow

PORT = os.getenv("PORT")
TRACKER_PATH = os.getenv("TRACKER_PATH") 
ARTIFACT_DIR="server"
app = FastAPI()

class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout: int):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        try:
            # Set the timeout for the request
            async with async_timeout.timeout(self.timeout):
                response = await call_next(request)
                return response
        except asyncio.TimeoutError:
            return JSONResponse(
                {"detail": "Request timeout exceeded"}, status_code=504
            )

app = FastAPI()
# Add the middleware to your FastAPI application
app.add_middleware(TimeoutMiddleware, timeout=3) 

@app.on_event("startup")
async def startup_event():
    (MlFlowManagement(TRACKER_PATH, "humamtest")
        .begin()
        .set_client()
        .set_artifact_destination(ARTIFACT_DIR)
        .load_all_artifacts()
    )

@app.get("/health")
async def health():
    return {
        "message": "ok"
    }

@app.get("/cfs2017")
async def cfs2017(request: Request):
    raw: Dict[str, str] = dict(request.query_params)
    missing_variable = precheck(raw)
    if len(missing_variable) > 0:
        return JSONResponse(
            status_code=401,
            content={
                "message": "lack of required variable",
                "data": missing_variable
            }
        )
    transformed = transform(raw)
    cost = infer(transformed)
    return JSONResponse(
        status_code=200,
        content={
            "message": "success",
            "data": {
                "cost": cost,
            }
        }
    )

# TODO move transofrmation to MLFlow Management, endpoint should only use it
def transform(raw):
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
        new_value = implement_preprocess(mapp[key], value)
        container.append(new_value)
    return np.concat(container, axis=1)

def implement_preprocess(col, original_value):
    # TODO it need loaded once when the handler start running
    # so any request just use loaded pickle; not reading from disk
    directory_path = f"{ARTIFACT_DIR}/artifacts/preprocess/{col}"
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    # if there is no files, it means there are no transformation for that column.
    # so we just return the original value
    for file in files:
        with open(f"{directory_path}/{file}", 'rb') as f:
            encoder = pickle.load(f)
            return handle_encoder(encoder, original_value)
    return original_value

def handle_encoder(encoder, value):
    return encoder.transform(np.array(value).reshape(1, -1).astype(np.int64))


def precheck(raw):
    important = [
        "origin_state",
        "destination_state",
        "naics",
        "mode",
        "shipment_weight",
        "shipment_distance_route",
    ]
    not_exist_key = []
    for i in important:
        if i not in raw.keys():
            not_exist_key.append(i) 
    return not_exist_key

MODEL = None
def infer(iinput):
    global MODEL
    if MODEL is None:
        directory_path = f"{ARTIFACT_DIR}/artifacts/cfs_model/model.pkl"
        with open(directory_path, 'rb') as f:
            model = pickle.load(f)
            MODEL = model
            return model.predict(iinput)[0]
    return MODEL.predict(iinput)[0]

# MLFlow class handling loading artifacts for server.
class MlFlowManagement:
    def __init__(self, tracking_url, name):
        self.tracking_url = tracking_url
        self.experiment = name
        self.client = None

    def begin(self):
        mlflow.set_tracking_uri(uri=self.tracking_url)
        mlflow.set_experiment(self.experiment)
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
        mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=f"{self.artifact_destination}")
        return self
