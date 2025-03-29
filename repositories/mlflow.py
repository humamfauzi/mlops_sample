import mlflow
import json
from abc import ABC
import random
import pickle
from mlflow.tracking import MlflowClient

from repositories.abc import Manifest
def generate_random_string(length: int) -> str:
    char = "ABCDEFGHIJKLMOPQRSTUVWXYZ1234567890"
    final = ""
    for _ in range(length):
        final += random.choice(char)
    return final


TEMP_DIR = "/tmp/mlflow/artifacts"

class MLflowRepository(Manifest):
    """
    Remote Repository Directory Structure:

    The remote repository stores models, transformations, and metadata, with their manifests, 
    facilitating versioning and retrieval. All of this under experiement name and parent run id

    model
    ├── MOD-001                   
    │   └── model.pkl             
    └── MOD-00N                   
        └── model.pkl             
    model_manifest.json           
    transformation
    ├── ColumnName1               
    │   ├── Transformation_name1  
    │   │   ├── transformation.pkl
    │   │   └── inverse.pkl       
    │   └── Transformation_name2  
    │       ├── transformation.pkl
    │       └── inverse.pkl       
    └── ColumnNameN               
        └── Transformation_name   
            ├── transformation.pkl
            └── inverse.pkl       
    transformation_manifest.json  
    metadata.json                 

    Note: in local we store it in the base path
    """
    def __init__(self, tracker_path: str, experiment_name: str, base_path: str):
        """
        Initializes the MLflowRepository class. 
        This handles all interactions with MLflow for logging and retrieving artifacts.

        :param tracker_path: The URI for the MLflow tracking server.
        :param experiment_name: The name of the MLflow experiment.
        :param base_path: The base path for saving artifacts.

        Note: This means no other instance can called mlflow except on this class
        """
        mlflow.set_tracking_uri(uri=tracker_path)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        self.base_path = base_path

    def start(self):
        """
        Starts a new MLflow run.
        :return: The active run ID.
        """
        self.active_run = mlflow.start_run(run_name=generate_random_string(6))
        return self.active_run.info.run_id

    def stop(self):
        """
        Stops the current MLflow run.
        """
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def get_parent_run_id(self):
        """
        Retrieves the active MLflow run.
        :return: The active run object.
        """
        return mlflow.active_run().info.run_id

    def get_experiment_id(self):
        """
        Retrieves the experiment ID for the current MLflow experiment.
        :return: The experiment ID.
        """
        return self.experiment_id

    def get_model_manifest(self, id: str):
        """
        Retrieves the model manifest from MLflow.
        :param id: The unique identifier for the model (e.g., model version).
        :return: The model manifest as a dictionary, or None if not found.
        """
        try:
            # Assuming the manifest is logged as a JSON artifact
            manifest_path = f"model_manifest_{id}.json"
            local_path = mlflow.get_artifact_uri(artifact_path=manifest_path)
            if local_path:
                with open(local_path, "r") as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error retrieving model manifest: {e}")
            return None

    def get_transformation_manifest(self, id: str):
        """
        Retrieves the transformation manifest from MLflow.
        :param id: The unique identifier for the transformation.
        :return: The transformation manifest as a dictionary, or None if not found.
        """
        try:
            # Assuming the manifest is logged as a JSON artifact
            manifest_path = f"transformation_manifest_{id}.json"
            local_path = mlflow.get_artifact_uri(artifact_path=manifest_path)
            if local_path:
                with open(local_path, "r") as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error retrieving transformation manifest: {e}")
            return None

    def get_metadata_manifest(self, id: str):
        """
        Retrieves the metadata manifest from MLflow.
        :param id: The unique identifier for the metadata.
        :return: The metadata manifest as a dictionary, or None if not found.
        """
        try:
            manifest_path = f"metadata_manifest_{id}.json"
            local_path = mlflow.get_artifact_uri(artifact_path=manifest_path)
            if local_path:
                with open(local_path, "r") as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error retrieving metadata manifest: {e}")
            return None

    def save_model_manifest(self, id: str, manifest: dict):
        """
        Saves the model manifest to MLflow as a JSON artifact.
        :param id: The unique identifier for the model.
        :param manifest: The model manifest to be saved (dictionary).
        """
        try:
            manifest_path = f"model_manifest_{id}.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
            mlflow.log_artifact(manifest_path)
        except Exception as e:
            print(f"Error saving model manifest: {e}")

    def save_transformation_manifest(self, id: str, manifest: dict):
        """
        Saves the transformation manifest to MLflow as a JSON artifact.
        :param id: The unique identifier for the transformation.
        :param manifest: The transformation manifest to be saved (dictionary).
        """
        try:
            manifest_path = f"transformation_manifest_{id}.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
            mlflow.log_artifact(manifest_path)
        except Exception as e:
            print(f"Error saving transformation manifest: {e}")

    def save_metadata_manifest(self, id: str, manifest: dict):
        """
        Saves the metadata manifest to MLflow as a JSON artifact.
        :param id: The unique identifier for the metadata.
        :param manifest: The metadata manifest to be saved (dictionary).
        """
        try:
            manifest_path = f"metadata_manifest_{id}.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
            mlflow.log_artifact(manifest_path)
        except Exception as e:
            print(f"Error saving metadata manifest: {e}")

    def save_folder(self, path: str):
        run_id = self.get_parent_run_id()
        mlflow.log_artifacts(
            run_id=run_id, 
            local_dir=f"{self.base_path}/{path}", 
            artifact_path=path
        )

    def load_model(self, run_id: str):
        """
        Loads a model from the specified path.
        :param path: The path to the model file.
        :return: The loaded model.
        """

    def compose_model_path(self, child_run_name: str):
        run_id = self.get_parent_run_id()
        return f"mlflow-artifacts:/{self.experiment_id}/{run_id}/artfacts/{child_run_name}/model.pkl"

    def compose_transformation_path(self, column_name: str, transformation_name: str, is_inverse: bool = False):
        run_id = self.get_parent_run_id()
        if is_inverse:
            return f"mlflow-artifacts:/{self.experiment_id}/{run_id}/artfacts/{column_name}/{transformation_name}/inverse.pkl"
        return f"mlflow-artifacts:/{self.experiment_id}/{run_id}/artfacts/{column_name}/{transformation_name}/transformation.pkl"

    def load_data_temporary(self, path: str):
        mlflow.artifacts.download_artifacts(artifact_uri=path, dst_path=TEMP_DIR)
        pass

    def load_model(self, path: str):
        mlflow.artifacts.download_artifacts(artifact_uri=path, dst_path=TEMP_DIR)
        with open(f"{TEMP_DIR}/model.pkl", 'rb') as f:
            model = pickle.load(f)
            return model

    def save_data(self, path: str, data: dict):
        pass

    def find_child_runs(self, order_by: str, filt: dict):
        """
        Finds a child run of the given run ID.
        :param run_id: The parent run ID.
        :return: The child run object, or None if not found.
        """
        filt["mlflow.parentRunId"] = self.get_parent_run_id()
        filter_string = self._compose_filter_string(filt)
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id], 
            filter_string=filter_string, 
            order_by=[order_by])
        return runs

    def set_tag(self, iden: str, key: str, value: any):
        """
        Sets a tag for the given run ID.
        :param id: The run ID.
        :param tag: The tag to be set (dictionary).
        """
        client = MlflowClient()
        client.set_tag(iden, key=key, value=value)

    def _compose_filter_string(self, ddict):
        final = []
        for key, value in ddict.items():
            final.append(f"tags.{key} = '{value}'")
        return str(" AND ".join(final))


    