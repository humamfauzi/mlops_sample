from typing import List
from repositories.struct import TransformationInstruction, TransformationObject, ModelObject

class Repository:
    """Repository stub that silently ignores all write operations."""

    def new_experiment(self, name: str):
        return "test"

    def list_all_experiments(self):
        return []

    def new_run(self, name: str, experiment_id: str):
        return 1

    def new_child_run(self, name: str, parent_run_id: int, experiment_id: str):
        return 2

    def new_property(self, run_id: int, key: str, value: str):
        return None

    def new_metric(self, run_id: int, key: str, value: float):
        return None

    def new_object(self, run_id: int, type: str, url: str):
        return None

    def find_best_model_within_run(self, parent_run_id: int, metric: str):
        return None

    def upsert_tag(self, run_id: int, key: str, value: str):
        return None

    def find_all_available_runs(self, experiment_id: str):
        return []

    def find_property(self, run_id: int, key: str):
        return None

    def get_all_published_candidates(self, experiment_id: str):
        return []

    def get_intent(self, run_id: int):
        return None

    def get_model_run_id(self, model_id: str) -> tuple[int, str]:
        return 0, ""

    def select_previously_published(self, experiment_id: str, intent: str, primary_metric: str):
        return None, None

class Object:
    def save_transformation_instruction(self, run_id: int, instructions: List[TransformationInstruction]):
        return None

    def save_transformation_object(self, run_id: int, transformation_objects: List[TransformationObject]):
        return None

    def load_transformation_instruction(self, run_id: int):
        return None

    def load_transformation_object(self, run_id: int):
        return None

    def save_model(self, run_id: int, model: ModelObject):
        return None

    def load_model(self, id: str):
        return None
        


