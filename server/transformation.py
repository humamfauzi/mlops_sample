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
        instructions = facade.load_transformation_instruction(run_id)
        transformations, itransformations = [], []

        for step in instructions:
            print(step)
            fobject = facade.load_transformation_object(step.id)
            def fn(input):
                selected = input[step.column]
                selected = fobject.transform(selected)
                input[step.column] = selected
                return input
            transformations.append(fn)
            if step.inverse_transform:
                def ifn(output):
                    selected = output[step.column]
                    selected = fobject.inverse_transform(selected)
                    output[step.column] = selected
                    return output
                itransformations.append(ifn)

        t = cls(facade)
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

