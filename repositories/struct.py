from dataclasses import dataclass
from typing import List
from enum import Enum

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