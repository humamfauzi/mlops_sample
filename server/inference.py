
from repositories.repo import Facade
class InferenceManager:
    """
    Primary class for handling inference requests
    Contain both transformation and models. This is the only class
    that called from FastAPI endpoint.
    
    One experiment should be managed by one InferenceManager instance.
    """
    def __init__(self, repository: Facade, experiment_id: str):
        self.repository: Facade = repository
        self.inferences = {}
        self.experiment_id = experiment_id

    @classmethod
    def parse_instruction(cls, repository: Facade, config: dict):
        experiment_id = config.get("experiment_id", "sample")
        pm = repository.get_all_published_candidates(experiment_id)
        c = cls(repository, experiment_id)
        for m in pm:
            name, id = m["name"], m["id"]
            c.inferences[name] = Inference.parse_instruction(repository, id, name)
        return c

    def infer(self, model_name: str, input_data: dict):
        infer = self.inferences.get(model_name, None)
        if infer is None:
            raise ValueError(f"Model {model_name} not found")
        return infer.infer(input_data)

    def list(self):
        return [inf.short_description() for inf in self.inferences.values()]

    def get_model_by_name(self, model_name: str):
        return self.inferences.get(model_name, None)

    def metadata(self, model_name: str):
        infer = self.inferences.get(model_name, None)
        if infer is None:
            raise ValueError(f"Model {model_name} not found")
        return infer.metadata()

from server.transformation import Transformation
from server.model import Model
class Inference:
    def __init__(self, transformation: Transformation, model: Model, description: str = "", name: str = ""):
        self.transformation: Transformation = transformation
        self.model: Model = model
        self.description = description
        # reperesent the main run name not the model name.
        self.name = name
    
    def infer(self, data: dict):
        data = self.transformation.parse_input(data)
        transformed = self.transformation.transform(data)
        result = self.model.infer(transformed)
        inverse = self.transformation.inverse_transform(result)
        return inverse

    @classmethod
    def parse_instruction(cls, repository: Facade, run_id: int, name: str = ""):
        desc = repository.get_run_description(run_id)
        transformation = Transformation.construct(repository, run_id)
        model = Model.construct(repository, run_id)
        return cls(transformation, model, desc, name)

    def short_description(self):
        return {
            "main_id": self.name,
            "model_id": self.model.filename,
            "description": self.description,
            "input": [inp for inp in self.transformation.get_available_input()],
        }

    def metadata(self):
        return {
            "description": self.description,
            "input": [inp for inp in self.transformation.get_available_input()],
        }

    

        