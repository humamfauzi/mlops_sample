
from repositories.repo import Facade 

class Model:
    def __init__(self, repository: Facade, experiment_id: str, model, name: str):
        self.repository = repository
        self.experiment_id = experiment_id
        self.cache = {}
        self.model = model
        self.name = name

    @classmethod
    def construct(cls, repository: Facade, run_id: int):
        model = repository.get_model_best_model(run_id)
        c = cls(repository, run_id, model.object, model.filename)
        return c

    def infer(self, data):
        print(">>>>>", data)
        result = self.model.predict(data)
        return result