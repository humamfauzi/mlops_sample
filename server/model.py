
from repositories.repo import Facade 

class Model:
    def __init__(self, repository: Facade, experiment_id: str):
        self.repository = repository
        self.experiment_id = experiment_id
        self.cache = {}

    @classmethod
    def construct(self, repository: Facade, run_id: int):
        model = repository.get_model_best_model(run_id)
        return model





