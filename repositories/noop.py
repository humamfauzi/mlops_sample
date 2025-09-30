
class Repository:
    """Repository stub that silently ignores all write operations."""

    def new_experiment(self, name: str):
        return "test"

    def list_all_experiments(self):
        return []

    def new_run(self, name: str, experiment_id: str):
        return 1

    def new_property(self, run_id: int, key: str, value: str):
        return None

    def new_metric(self, run_id: int, key: str, value: float):
        return None

    def new_object(self, run_id: int, type: str, url: str):
        return None

class Object:
    def save_transformation_instruction(self, instructions: List[TransformationInstruction]):
        return None

    def save_transformation_object(self, transformation_objects: List[TransformationObject]):
        return None

    def load_transformation_instruction(self):
        return None

    def load_transformation_object(self):
        return None

    def save_model(self, model: ModelObject):
        return None

    def load_model(self, id: str):
        return None
        


