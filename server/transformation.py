from repositories.repo import Facade
import pandas as pd
import numpy as np
from enum import Enum

# TODO: Should be generalized for both train and serve modules
class TransformationMethods(Enum):
    # would replace the original column with the transformation
    REPLACE = 1
    # would append the transformation to the original column; the original
    # column would still exist
    APPEND = 2

    # would append the transformation to the original column and remove the original
    APPEND_AND_REMOVE = 3

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
            print(step)
            fobject = facade.load_transformation_object(run_id, step.id)
            if step.inverse_transform:
                func = getattr(fobject.object, "inverse_transform", None)
                if callable(func):
                    itransformations.append({
                        "name": step.name,
                        "column": step.column,
                        "method": step.method,
                        "function": func
                    })
            else:
                func = getattr(fobject.object, "transform", None)
                get_feature_names_out = getattr(fobject.object, "get_feature_names_out", None)
                if callable(func):
                    transformations.append({
                        "name": step.name,
                        "column": step.column,
                        "method": step.method,
                        "function": func,
                        "feature_names": get_feature_names_out,
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
        print("before", transformed.to_dict())
        for transformation in self.transformations:
            column = transformation["column"]
            if column in self.column.categorical():
                input = transformed[column].astype(str)
            else:
                input = transformed[column].astype(float)
            method = transformation.get("method", "")
            if method == TransformationMethods.REPLACE.name:
                transformed[column] = transformation["function"](input.to_numpy().reshape(-1, 1))
            elif method == TransformationMethods.APPEND.name:
                appended = transformation["function"](input.to_numpy().reshape(-1, 1))
                transformed[f"{column}_{transformation['name']}"] = appended
            elif method == TransformationMethods.APPEND_AND_REMOVE.name:
                appended = transformation["function"](input.to_numpy().reshape(-1, 1))
                encoded_columns = transformation["feature_names"]([column])
                new_columns = pd.DataFrame(appended, columns=encoded_columns, dtype='int', index=transformed.index)
                transformed = pd.concat([transformed.drop(column, axis=1), new_columns], axis=1)
        print("after", transformed.to_dict())
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

