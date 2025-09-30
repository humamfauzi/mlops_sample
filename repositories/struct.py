from dataclasses import dataclass
from typing import List
from enum import Enum

class TransformationInstruction:
    id: str
    name: str
    columns: List[Enum]
    function: any
    method: Enum

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "columns": self.columns,
            "function": self.function,
            "method": self.method,
        }

class TransformationObject:
    filename: str
    object: any

class ModelObject:
    filename: str
    object: any