import asyncio
from fastapi import FastAPI, Request 
from fastapi.middleware.cors import CORSMiddleware

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import Optional
from contextlib import asynccontextmanager
from repositories.repo import Facade
from dotenv import load_dotenv, find_dotenv

import buildinfo
import runtime_config
import server.response as response
from server.inference import InferenceManager
from server.error import UserError

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

def load_settings() -> dict:
    """Resolve repository settings from .env layered over the shared config.

    The trainer resolves the same structure through the same module, so the two
    runtimes cannot disagree about which database they are using.
    """
    try:
        dotenv_file = find_dotenv(usecwd=True)
        if dotenv_file:
            load_dotenv(dotenv_file, override=False)
    except Exception as e:
        raise ValueError("Failed to load .env file") from e
    return runtime_config.load()


@asynccontextmanager
async def lifespan(app):
    global model
    settings = load_settings()
    # Log the resolved repository so a misconfigured deployment is obvious in
    # the startup log rather than showing up as a mysterious empty model list.
    # flush=True: stdout is block-buffered when the server runs under a process
    # manager, so without it this line is lost on shutdown.
    print(f"runtime config: {runtime_config.describe()}", flush=True)
    repo = Facade.parse_instruction(settings)
    model = InferenceManager.parse_instruction(repo, {
        "experiment_id": settings["experiment_id"],
        "column_reference": settings["column_reference"],
    })
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
    build = buildinfo.describe()
    config = runtime_config.describe()
    if model is None:
        return response.HealthResponse(
            status="unavailable",
            build=build,
            config=config,
            http_status=503,
        ).to_json_response()
    h = model.health()
    if h["loaded"] == 0:
        return response.HealthResponse(
            status="unavailable",
            model_count=0,
            failed=h["failed"],
            failures=h["failures"],
            build=build,
            config=config,
            http_status=503,
        ).to_json_response()
    return response.HealthResponse(
        status="ok" if h["failed"] == 0 else "degraded",
        model_count=h["loaded"],
        failed=h["failed"],
        failures=h["failures"],
        build=build,
        config=config,
        http_status=200,
    ).to_json_response()

# Primary endpoint for getting all available model for CFS 2017 problems
@app.get("/cfs2017")
async def cfs2017():
    all_model = model.list()
    return response.ListResponse(message="success", data=all_model).to_json_response()

@app.get("/cfs2017/enum_maps")
async def cfs2017EnumMaps():
    enum_maps = model.enum_maps()
    return response.EnumMapsResponse(message="success", enum_maps=enum_maps).to_json_response()

# Primary endpoint for getting all metadata about the model
@app.get("/cfs2017/{model}/metadata")
async def cfs2017ModelMetadata(request: Request):
    name = request.path_params["model"]
    dd = model.metadata(model_name=name)
    return response.MetadataReponse(
        message="success", 
        metadata=dd["metadata"], 
        input=dd["input"],
        description=dd["description"]
    ).to_json_response()

# Primary endpoint for inference using the model
@app.get("/cfs2017/{model}/inference")
async def cfs2017ModelInference(request: Request):
    req_model = request.path_params["model"]
    # filter first so any thing that goes to inference model is only
    # what the model requires and discard any extra input
    result = model.infer(req_model, dict(request.query_params))
    return response.InferenceResponse(message="success", output=result).to_json_response()

@app.exception_handler(UserError)
async def user_error_handler(_: Request, exc: UserError):
    return JSONResponse(
        status_code=exc.http_code,
        content=exc.to_dict()
    )