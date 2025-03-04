from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from starlette.responses import JSONResponse
import server.model as model

# How to use the data class for representing response message
# @app.get("/cfs2017")
# async def cfs2017(request: Request):
#     models = [
#         ModelItem(value="MOD-001", display="Dummy Model 1"),
#         ModelItem(value="MOD-002", display="Dummy Model 2")
#     ]
#     response = ModelListResponse(message="", data=models)
#     return JSONResponse(
#         status_code=200,
#         content=response.to_dict()
#     )
# 
# @app.get("/cfs2017/{model}/metadata")
# async def cfs2017ModelMetadata(request: Request):
#     model = request.path_params["model"]
#     
#     metadata = [
#         MetadataItem(key="name", display="Model Name", value="Dummy Model"),
#         MetadataItem(key="train_accuracy", display="Train Accuracy", value=0.8)
#         # Add more metadata items as needed
#     ]
#     
#     inputs = [
#         InputItem(type="categorical", key="origin", display="Origin", 
#                  enumeration=["USA", "Europe", "Japan"]),
#         # Add more input items as needed
#     ]
#     
#     content = ModelMetadataContent(
#         description="This is a dummy model. Use this structure as a reference.",
#         metadata=metadata,
#         input=inputs
#     )
#     
#     response = ModelMetadataResponse(message="", data=content)
#     return JSONResponse(
#         status_code=200,
#         content=response.to_dict()
#     )

@dataclass
class ListResponse:
    """Response structure for the cfs2017 endpoint"""
    message: str
    data: List[model.ShortDescription] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "data": [item.to_dict() for item in self.data]
        }
    def to_json_response(self) -> JSONResponse:
        return JSONResponse(status_code=200, content=self.to_dict())


@dataclass
class MetadataReponse:
    """Content structure for model metadata"""
    message: str
    metadata: List[model.Metadata]
    input: List[model.Input]
    description: str
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "data": {
                "description": self.description,
                "metadata": [item.to_dict() for item in self.metadata],
                "input": [item.to_dict() for item in self.input]
            }
        }

    def to_json_response(self) -> JSONResponse:
        return JSONResponse(status_code=200, content=self.to_dict())

@dataclass
class InferenceResponse:
    """Content structure for inference response"""
    message: str
    output: model.Output
    def to_dict(self) -> Dict[str, str]:
        return {
            "message": self.message,
            "data": self.output.to_dict()
        }
    def to_json_response(self) -> JSONResponse:
        return JSONResponse(status_code=200, content=self.to_dict())

class ErrorResponse:
    code: int
    message: str
    information: Dict[str, Any]
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "data": {
                "information": self.information
            }
        }
    def to_json_response(self) -> JSONResponse:
        return JSONResponse(status_code=self.code, content=self.to_dict())