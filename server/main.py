import asyncio
import os
from fastapi import FastAPI, Request 
from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from server.inference import InferenceManager
import server.response as response
from typing import Optional
from contextlib import asynccontextmanager
from repositories.repo import Facade
from dotenv import load_dotenv, find_dotenv

origins = [
    "http://localhost:3000",
    "https://humamf.com"
]

class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout: int):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next):
        try:
            response = await asyncio.wait_for(call_next(request), timeout=self.timeout)
            return response
        except asyncio.TimeoutError:
            return JSONResponse(
                {"detail": "Request timeout exceeded"}, status_code=504
            )

# a singleton model provider
# the model would be loaded. Once loaded, it would be accessed via API
model: Optional[InferenceManager] = None

def load_env() -> dict:
    try:
        dotenv_file = find_dotenv(usecwd=True)
        if dotenv_file:
            load_dotenv(dotenv_file, override=False)
    except Exception as e:
        raise ValueError("Failed to load .env file") from e
    instruction = {
        "experiment_id": os.getenv("EXPERIMENT_ID", "sample"),
        "stage": os.getenv("STAGE", "dev"),
        "data": {
            "type": os.getenv("REPOSITORY_DATA", "sqlite"),
            "properties": {
                "name": os.getenv("REPOSITORY_DATA_PATH", "example.db")
            }
        },
        "object": {
            "type": os.getenv("REPOSITORY_OBJECT", "s3"),
            "properties": {
                "bucket_name": os.getenv("REPOSITORY_S3_BUCKET", "humamf-artifacts")
            }
        }
    }
    return instruction

@asynccontextmanager
async def lifespan(app):
    global model
    instruction = load_env()
    repo = Facade.parse_instruction(instruction)
    model = InferenceManager.parse_instruction(repo, {"experiment_id": instruction["experiment_id"]})
    try:
        yield
    finally:
        # optional cleanup on shutdown
        model = None

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add timeout middleware to limit the request
app.add_middleware(TimeoutMiddleware, timeout=3) 
lifespan(app)

@app.get("/health")
async def health():
    return response.HealthResponse(status="ok").to_json_response()

# Primary endpoint for getting all available model for CFS 2017 problems
@app.get("/cfs2017")
async def cfs2017():
    all_model = model.list()
    return response.ListResponse(message="success", data=all_model).to_json_response()

# Primary endpoint for getting all metadata about the model
@app.get("/cfs2017/{model}/metadata")
async def cfs2017ModelMetadata(request: Request):
    name = request.path_params["model"]
    dd = model.metadata(model_name=name)
    return response.MetadataReponse(
        message="success", 
        metadata={},
        # metadata=dd["metadata"], 
        input=dd["input"], 
        description=dd["description"]
    ).to_json_response()

# Primary endpoint for inference using the model
@app.get("/cfs2017/{model}/inference")
async def cfs2017ModelInference(request: Request):
    req_model = request.path_params["model"]
    # filter first so any thing that goes to inference model is only
    # what the model requires and discard any extra input
    result = model.infer(req_model, request.query_params)
    return response.InferenceResponse(message="success", output=result).to_json_response()
