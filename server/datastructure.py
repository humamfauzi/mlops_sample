from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

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
class BaseResponse:
    """Base class for all API responses"""
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONResponse"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (BaseResponse, ModelItem, MetadataItem, InputItem)):
                result[key] = value.to_dict()
            elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'to_dict'):
                result[key] = [item.to_dict() for item in value]
            else:
                result[key] = value
        return result


@dataclass
class ModelItem:
    """Model item for the model list response"""
    value: str
    display: str
    
    def to_dict(self) -> Dict[str, str]:
        return { "value": self.value, "display": self.display }


@dataclass
class ModelListResponse(BaseResponse):
    """Response structure for the cfs2017 endpoint"""
    data: List[ModelItem] = field(default_factory=list)


@dataclass
class MetadataItem:
    """Metadata item for the model metadata response"""
    key: str
    display: str
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "display": self.display,
            "value": self.value
        }


@dataclass
class InputItemBase:
    """Base class for input items"""
    key: str
    display: str


@dataclass
class CategoricalInputItem(InputItemBase):
    """Categorical input item for the model metadata response"""
    type: str = "categorical"
    enumeration: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "key": self.key,
            "display": self.display,
            "enumeration": self.enumeration
        }


@dataclass
class NumericalInputItem(InputItemBase):
    """Numerical input item for the model metadata response"""
    type: str = "numerical"
    min: float = None
    max: float = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type,
            "key": self.key,
            "display": self.display
        }
        if self.min is not None:
            result["min"] = self.min
        if self.max is not None:
            result["max"] = self.max
        return result


InputItem = Union[CategoricalInputItem, NumericalInputItem]


@dataclass
class ModelMetadataContent:
    """Content structure for model metadata"""
    description: str
    metadata: List[MetadataItem]
    input: List[InputItem]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "metadata": [item.to_dict() for item in self.metadata],
            "input": [item.to_dict() for item in self.input]
        }


@dataclass
class ModelMetadataResponse(BaseResponse):
    """Response structure for the model metadata endpoint"""
    data: ModelMetadataContent = None


@dataclass
class InferenceResponseContent:
    """Content structure for inference response"""
    result: str
    inference_time: str
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "result": self.result,
            "inference_time": self.inference_time
        }


@dataclass
class InferenceResponse(BaseResponse):
    """Response structure for the inference endpoint"""
    data: InferenceResponseContent = None


@dataclass
class ErrorData:
    """Error data structure"""
    stack: List[str] = field(default_factory=list)
    information: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stack": self.stack,
            "information": self.information
        }


@dataclass
class ErrorResponse(BaseResponse):
    """Response structure for errors"""
    data: ErrorData = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = ErrorData()