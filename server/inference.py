
from repositories.repo import Facade
class InferenceManager:
    """
    Primary class for handling inference requests
    Contain both transformation and models. This is the only class
    that called from FastAPI endpoint.
    """
    def __init__(self, repository: Facade):
        self.repository: Facade = repository
        self.inferences = {}
        pass

    @classmethod
    def parse_instruction(cls, repository: Facade, config: dict):
        pm = repository.get_all_published_models(config.get("experiment_id", "sample"))
        c = cls(repository)
        for m in pm:
            c.inferences[m.name] = Inference.parse_instruction(repository, m)
        return c

    def get_inference(self, model_id: str, input_data: dict):
        infer = self.inferences.get(model_id, None)
        if infer is None:
            raise ValueError(f"Model {model_id} not found")
        return infer.infer(input_data)

from server.transformation import Transformation
from server.model import Model
class Inference:
    def __init__(self, transformation: Transformation, model: Model):
        self.transformation: Transformation = transformation
        self.model: Model = model

    def infer(self, data: dict):
        data = self.transformation.parse_input(data)
        transformed = self.transformation.transform(data)
        result = self.model.infer(transformed)
        inverse = self.transformation.inverse_transform(result)
        return inverse

    @classmethod
    def parse_instruction(cls, repository: Facade, run_id: int):
        transformation = Transformation.construct(repository, run_id)
        model = Model.construct(repository, run_id)
        return cls(transformation, model)

    def short_description(self):
        return {
            "model_id": self.model.experiment_id,
            "description": self.model.description,
            "input": [inp.to_dict() for inp in self.transformation.input],
            "metadata": [meta.to_dict() for meta in self.model.metadata]
        }
        