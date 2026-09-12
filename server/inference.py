
from repositories.repo import Facade
from column.cfs2017 import TabularColumn # TODO: should move this out of cfs2017 file
from typing import Dict
from server.enum_maps import primary
from server.error import UserError
import traceback

class InferenceManager:
    """
    Primary class for handling inference requests
    Contain both transformation and models. This is the only class
    that called from FastAPI endpoint.
    
    One experiment should be managed by one InferenceManager instance.
    """
    def __init__(self, repository: Facade, experiment_id: str, column_reference: TabularColumn):
        self.repository: Facade = repository
        self.inferences: Dict[str, Inference] = {}
        self.experiment_id = experiment_id
        self.column_reference = column_reference
        # Models that were published but failed to load, keyed by run name.
        # Surfaced through health() instead of being silently dropped.
        self.failures: Dict[str, str] = {}

    @classmethod
    def parse_instruction(cls, repository: Facade, config: dict):
        experiment_id = config.get("experiment_id", "sample")
        pm = repository.get_all_published_candidates(experiment_id)
        column_reference = TabularColumn.from_string(config.get("column_reference", "sample"))
        c = cls(repository, experiment_id, column_reference)
        for m in pm:
            name, id = m["name"], m["id"]
            try:
                c.inferences[name] = Inference.parse_instruction(repository, id, name, column_reference)
            except Exception as e:
                c.failures[name] = f"{type(e).__name__}: {e}"
                print(f"Error parsing instruction for model {name}: {e}")
                traceback.print_exc()

        # A server that loaded nothing is not usable. Raise rather than come up
        # healthy and 404/500 on every request.
        if pm and not c.inferences:
            detail = "; ".join(f"{k} -> {v}" for k, v in c.failures.items())
            raise RuntimeError(
                f"No models could be loaded for experiment '{experiment_id}' "
                f"({len(pm)} published candidate(s), all failed): {detail}"
            )
        return c

    def health(self) -> dict:
        return {
            "loaded": len(self.inferences),
            "failed": len(self.failures),
            "models": sorted(self.inferences.keys()),
            "failures": self.failures,
        }

    def infer(self, model_name: str, input_data: dict):
        infer = self.inferences.get(model_name, None)
        if infer is None:
            raise UserError(f"Model '{model_name}' not found") \
                .set_kvs({"model": model_name, "available_models": sorted(self.inferences.keys())}) \
                .set_http_status(404)
        return {
            self.column_reference.target().lower(): infer.infer(input_data)
        }

    def list(self):
        return [inf.short_description() for inf in self.inferences.values()]

    def enum_maps(self):
        return primary

    def get_model_by_name(self, model_name: str):
        return self.inferences.get(model_name, None)

    def metadata(self, model_name: str):
        infer = self.inferences.get(model_name, None)
        if infer is None:
            raise UserError(f"Model '{model_name}' not found") \
                .set_kvs({"model": model_name, "available_models": sorted(self.inferences.keys())}) \
                .set_http_status(404)
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
        return float(inverse[0])

    @classmethod
    def parse_instruction(cls, repository: Facade, run_id: int, name: str = "", column_reference: str = None):
        desc = repository.get_run_description(run_id)
        transformation = Transformation.construct(repository, run_id, column_reference)
        model = Model.construct(repository, run_id)
        return cls(transformation, model, desc, name)

    def short_description(self):
        desc = self.transformation.intent
        return {
            "main_id": self.name,
            "model_id": self.model.name,
            "description": desc,
            "input": [inp.to_dict() for inp in self.transformation.get_available_input()],
        }

    def metadata(self):
        return {
            "description": self.description,
            "input": [inp.to_dict() for inp in self.transformation.get_available_input()],
            "metadata": self.transformation.get_metadata(),
        }

    

        