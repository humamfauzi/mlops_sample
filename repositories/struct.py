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

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "column": self.column,
            "method": self.method,
            "inverse_transform": self.inverse_transform
        }


@dataclass
class TransformationObject:
    filename: str
    object: any

@dataclass
class ModelObject:
    filename: str
    object: any