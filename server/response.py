from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from starlette.responses import JSONResponse
from abc import ABC, abstractmethod
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

class Response(ABC):
    """Base class for all response classes"""
    @abstractmethod 
    def to_dict(self) -> Dict[str, Any]: pass
    '''
    Convert the response to a dictionary format.
    This method should be implemented by subclasses to provide
    '''

    @abstractmethod
    def to_json_response(self) -> JSONResponse: pass
    '''
    Convert the response to a JSONResponse format.
    This method should be implemented by subclasses to provide
    '''

@dataclass
class ListResponse(Response):
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

_ = ListResponse(message="", data=[])

@dataclass
class MetadataReponse(Response):
    """Content structure for model metadata"""
    message: str
    metadata: List[model.Metadata]
    input: List[model.Input]
    description: str
    def to_dict(self) -> Dict[str, Any]:
        print("metadata", self.metadata)
        print("input", self.input)
        return {
            "message": self.message,
            "data": {
                "description": self.description,
                "metadata": self.metadata,
                "input": self.input
            }
        }

    def to_json_response(self) -> JSONResponse:
        return JSONResponse(status_code=200, content=self.to_dict())

_ = MetadataReponse(message="", metadata=[], input=[], description="")
@dataclass
class InferenceResponse(Response):
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

_ = InferenceResponse(message="", output=model.NumericalInput(key="", display="", min=0.0, max=1.0))

@dataclass
class ErrorResponse(Response):
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


_ = ErrorResponse(code=500, message="", information={})