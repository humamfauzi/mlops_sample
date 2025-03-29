import async_timeout
import asyncio
import os
from types import Dict
from fastapi import FastAPI, Request, Query
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from server.model import ModelRepository
import server.response as response
from types import Dict

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
# Add timeout middleware to limit the request
app.add_middleware(TimeoutMiddleware, timeout=3) 

# a singleton model provider
CFS2017_MODEL_REPOSITORY = None
@app.on_event("startup")
async def startup_event():
    global CFS2017_MODEL_REPOSITORY
    CFS2017_MODEL_REPOSITORY = ModelRepository(TRACKER_PATH, "humamtest_lazycall").load()

# check the connection for the server
@app.get("/health")
async def health():
    return JSONResponse(content={"message": "ok"}, status_code=200)

# Primary endpoint for getting all available model for CFS 2017 problems
@app.get("/cfs2017")
async def cfs2017():
    all_model = CFS2017_MODEL_REPOSITORY.list()
    return response.ListResponse(message="success", data=all_model).to_json_response()

# Primary endpoint for getting all metadata about the model
@app.get("/cfs2017/{model}/metadata")
async def cfs2017ModelMetadata(request: Request):
    model = request.path_params["model"]
    metadata = CFS2017_MODEL_REPOSITORY.metadata(model_name=model)
    return response.MetadataReponse(message="success", data=metadata).to_json_response()

# Primary endpoint for inference using the model
@app.get("/cfs2017/{model}/inference")
async def cfs2017ModelInference(request: Request, q: Dict = Query(...)):
    model = request.path_params["model"]
    # filter first so any thing that goes to inference model is only
    # what the model requires and discard any extra input
    filtered, message = CFS2017_MODEL_REPOSITORY.validate_input(model, q)
    if message is not "":
        return response.ErrorResponse(code=401, message=message).to_json_response()
    result = CFS2017_MODEL_REPOSITORY.infer(model, filtered)
    return response.InferenceResponse(message="success", data=result).to_json_response()