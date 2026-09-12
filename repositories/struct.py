from dataclasses import dataclass
from typing import List
from enum import Enum

class TransformationMethods(Enum):
    """How a fitted transformation is applied to a column.

    Stored by name in the transformation instructions, so the training pipeline
    and the inference server must agree on these values. They are serialised as
    ``method.name`` and replayed by both `train/data_transform.py` and
    `server/transformation.py`, which is why the enum lives in the shared
    structures module rather than being defined in each.
    """
    # would replace the original column with the transformation
    REPLACE = 1
    # would append the transformation to the original column; the original
    # column would still exist
    APPEND = 2
    # would append the transformation to the original column and remove the original
    APPEND_AND_REMOVE = 3

@dataclass
class TransformationInstruction:
    id: str
    name: str
    column: str
    method: str
    inverse_transform: bool
    type: str

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "column": self.column,
            "method": self.method,
            "inverse_transform": self.inverse_transform,
            "type": self.type,
        }


@dataclass
class TransformationObject:
    filename: str
    object: any

@dataclass
class ModelObject:
    filename: str
    object: any