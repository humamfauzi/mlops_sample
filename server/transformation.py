from repositories.repo import Facade
import pandas as pd

class Transformation:
    def __init__(self, repository: Facade):
        self.repository: Facade = repository
        self.transformations = []
        self.itransformations = []
        self.available_input = []
        pass

    @classmethod
    def construct(cls, facade: Facade, run_id: int):
        transformations = []
        for step in instruction.get("transformation_steps", []):
            pass

        itransformations = []
        for step in instruction.get("inverse_transformation_steps", []):
            pass

        available_input = instruction.get("available_column", [])

        t = cls(facade)
        t.transformations = transformations
        t.itransformations = itransformations
        t.available_input = available_input
        return t

    def parse_input(self, input: dict) -> dict:
        for col in self.available_input:
            if col not in input:
                raise ValueError(f"Input data must contain column {col}")
        parsed = {k: v for k, v in input.items() if k in self.available_input}
        return parsed

    def transform(self, input: dict) -> pd.DataFrame:
        transformed = pd.DataFrame(input)
        for transformation in self.transformations:
            transformed = transformation(transformed)
        return transformed

    def inverse_transform(self, output):
        """
        Only exist for inference output only, so it has only one input
        """
        inversed = output
        for itransformation in self.itransformations:
            inversed = itransformation(inversed)
        return inversed

