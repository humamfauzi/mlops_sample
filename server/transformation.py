from repositories.repo import Facade
import pandas as pd
import numpy as np
from enum import Enum

class Transformation:
    def __init__(self, repository: Facade, column_reference: Enum):
        self.repository: Facade = repository
        self.transformations = []
        self.itransformations = []
        self.available_input = []
        self.column = column_reference
        pass

    @classmethod
    def construct(cls, facade: Facade, run_id: int, column_reference: Enum = None):
        instructions = facade.load_transformation_instruction(run_id)
        transformations, itransformations = [], []
        for step in instructions:
            fobject = facade.load_transformation_object(run_id, step.id)
            if step.inverse_transform:
                func = getattr(fobject.object, "inverse_transform", None)
                if callable(func):
                    itransformations.append({
                        "column": step.column,
                        "function": func
                    })
            else:
                func = getattr(fobject.object, "transform", None)
                if callable(func):
                    transformations.append({
                        "column": step.column,
                        "function": func
                    })
        t = cls(facade, column_reference)
        t.transformations = transformations
        t.itransformations = itransformations

        available_input = facade.get_available_input(run_id)
        t.available_input = available_input
        return t

    def get_available_input(self):
        return self.available_input

    def parse_input(self, input: dict) -> dict:
        for col in self.available_input:
            if col not in input:
                raise ValueError(f"Input data must contain column {col}")
        parsed = {k: v for k, v in input.items() if k in self.available_input}
        return parsed

    def transform(self, input: dict) -> pd.DataFrame:
        transformed = pd.DataFrame([input])
        for transformation in self.transformations:
            def transform(row):
                col = transformation["column"]
                input = row[col]
                if col in self.column.numerical():
                    input = float(input)
                input = np.array(input).reshape(-1, 1)
                print("Applying transformation on column:", col, "with input:", input)
                row[col] = transformation["function"](input)
                return row
            transformed = transformed.apply(transform, axis=1)
        return transformed

    def inverse_transform(self, output):
        """
        Only exist for inference output only, so it has only one input
        """
        inversed = output
        for itransformation in self.itransformations:
            # target can only have one column so we only use the function
            inversed = itransformation["function"](inversed)
        return inversed

