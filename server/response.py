from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from starlette.responses import JSONResponse
from abc import ABC, abstractmethod
from server.inference import Inference

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
class HealthResponse(Response):
    """Response structure for the health endpoint"""
    status: str
    def to_dict(self) -> Dict[str, str]:
        return {
            "status": self.status
        }
    def to_json_response(self) -> JSONResponse:
        return JSONResponse(status_code=200, content=self.to_dict())

@dataclass
class ListResponse(Response):
    """Response structure for the cfs2017 endpoint"""
    message: str
    data: List[dict] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "data": self.data
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