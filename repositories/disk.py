import os
from typing import List
import json
import pickle

from .struct import TransformationInstruction, TransformationObject, ModelObject

class Disk:
    def __init__(self, root: str):
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root)

    def save_transformation_instruction(self, run_id: int, instructions: List[TransformationInstruction]):
        all_instruction = [inst.to_dict() for inst in instructions]
        path = os.path.join(self.root, str(run_id), "transformation", "instruction.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_instruction, f, indent=2, ensure_ascii=False)
        return path


    def save_transformation_object(self, run_id: int, transformation_objects: List[TransformationObject]):
        trans_dir = os.path.join(self.root, str(run_id), "transformation", "objects")
        os.makedirs(trans_dir, exist_ok=True)

        filenames = []
        for obj in transformation_objects:
            file_path = os.path.join(trans_dir, obj.filename)
            with open(file_path, "wb") as f:
                pickle.dump(obj.object, f, protocol=pickle.HIGHEST_PROTOCOL)
            filenames.append(file_path)
        return filenames


    def load_transformation_instruction(self):
        path = os.path.join(self.root, "transformation", "instruction.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [TransformationInstruction(**item) for item in data]

    def load_transformation_object(self):
        path = os.path.join(self.root, "transformation", "objects")
        objects = []
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
                objects.append(TransformationObject(filename=filename, object=obj))
        return objects
        

    def save_model(self, run_id:int, model: ModelObject):
        os.makedirs(os.path.join(self.root, str(run_id), "model"), exist_ok=True)
        path = os.path.join(self.root, str(run_id), "model", f"{model.filename}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model.object, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def load_model(self, run_id: int, id: str):
        path = os.path.join(self.root, str(run_id), "model", f"{id}.pkl")
        with open(path, "rb") as f:
            model = pickle.load(f)
            return ModelObject(filename=f"{id}.pkl", object=model)
        


