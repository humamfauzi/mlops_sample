
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
    def parse_instruction(cls, config: dict):
        repository = Facade.parse_instruction(config.get("repository", {}))
        pm = repository.get_all_published_models()
        c = cls(repository)
        for m in pm:
            c.inferences[m.id] = Inference.parse_instruction(m)
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
        self.transformation = transformation
        self.model = model

    def infer(self, data: dict):
        transformed = self.transformation.transform(data)
        result = self.model.infer(transformed)
        inverse = self.transformation.inverse_transform(result)
        return inverse

    @classmethod
    def parse_instruction(cls, instruction):
        transformation = Transformation.parse_instruction(instruction.get("transformation", {}))
        model = Model.parse_instruction(instruction.get("model", {}))
        return cls(transformation, model)
        