import mlflow
import json
import random
import pickle
from typing import List
from mlflow.tracking import MlflowClient

from repositories.abc import Manifest
import os

def generate_random_string(length: int) -> str:
    char = "ABCDEFGHIJKLMOPQRSTUVWXYZ1234567890"
    final = ""
    for _ in range(length):
        final += random.choice(char)
    return final

TEMP_DIR = "/tmp/mlflow/artifacts"

class Tracker:
    """Handles MLflow metrics tracking."""

    def __init__(self, tracker_path: str, experiment_name: str):
        mlflow.set_tracking_uri(uri=tracker_path)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        self.active_run = None

    def log_metric(self, key: str, value: float):
        """Logs a metric to the current run."""
        mlflow.log_metric(key, value)

    def log_param(self, key: str, value: any):
        """Logs a parameter to the current run."""
        mlflow.log_param(key, value)

    def set_tag(self, key: str, value: any):
        """Sets a tag for the current run."""
        mlflow.set_tag(key, value)

    def set_tag_run(self, run_id: str, key: str, value: any):
        client = MlflowClient()
        client.set_tag(run_id, key, value)

    def get_active_run_id(self):
        """Returns the active run ID."""
        if self.active_run:
            return self.active_run.info.run_id
        return None

    def _compose_filter_string(self, filt: dict):
        """Composes a filter string for MLflow search."""
        filter_string = []
        for key, value in filt.items():
            filter_string.append(f"tags.{key}='{value}'")
        return " and ".join(filter_string)

    def find_child_runs(self, parent_run_id: str, order_by: str, filt: dict):
        filt["mlflow.parentRunId"] = parent_run_id
        filter_string = self._compose_filter_string(filt)
        print(f"Filter string: {filter_string}, {order_by}")
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id], 
            filter_string=filter_string, 
            order_by=[order_by])
        return runs

class Artifact:
    """Handles MLflow artifact storage."""

    def __init__(self, tracker_path: str, experiment_name: str):
        self.tracker_path = tracker_path
        self.experiment_name = experiment_name
        self.client = MlflowClient(tracking_uri=tracker_path)
        self.experiment_id = self._get_experiment_id()

    def _get_experiment_id(self):
        """Retrieves the experiment ID."""
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment:
            return experiment.experiment_id
        return None

    def save_transformation(self, func, parent_run_id: str, column: str, transformation_name: str):
        artifact_uri = f"transformation/{column}/{transformation_name}"
        local_path = f"{TEMP_DIR}/{artifact_uri}/transformation.pkl"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            pickle.dump(func, f)
        print(f"parent run {parent_run_id}")
        self.client.log_artifact(parent_run_id, local_path, artifact_uri)
        return

    def load_transformation(self, func, parent_run_id: str, column: str, transformation_name: str):
        artifact_uri = f"{parent_run_id}/transformation/{column}/{transformation_name}/transformation.pkl"
        local_path = f"{TEMP_DIR}/{artifact_uri}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{artifact_uri}",
            dst_path=local_path
        )
        with open(local_path, 'rb') as f:
            func = pickle.load(f)
            return func

    def save_transformation_manifest(self, manifest, parent_run_id: str):
        artifact_uri = f"transformation_manifest.json"
        local_path = f"{TEMP_DIR}/{artifact_uri}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'w') as f:
            json.dump(manifest, f)
        self.client.log_artifact(parent_run_id, local_path, artifact_uri)
        return

    def load_transformation_manifest(self, parent_run_id: str):
        artifact_uri = f"{parent_run_id}/transformation_manifest.json"
        local_path = f"{TEMP_DIR}/{artifact_uri}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{artifact_uri}", 
            dst_path=TEMP_DIR
        )
        with open(local_path, 'r') as f:
            manifest = json.load(f)
            return manifest

    def save_model_manifest(self, manifest, parent_run_id: str):
        artifact_uri = f"model_manifest.json"
        local_path = f"{TEMP_DIR}/{artifact_uri}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'w') as f:
            json.dump(manifest, f)
        self.client.log_artifact(parent_run_id, local_path, artifact_uri)

    def load_model_manifest(self, parent_run_id: str):
        artifact_uri = f"{parent_run_id}/model_manifest.json"
        local_path = f"{TEMP_DIR}/{artifact_uri}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{artifact_uri}", 
            dst_path=local_path
        )
        with open(local_path, 'r') as f:
            manifest = json.load(f)
            return manifest

    def save_model(self, model, parent_run_id: str, run_name: str):
        """Saves a model to MLflow."""
        artifact_uri = f"models/{run_name}"
        local_path = f"{TEMP_DIR}/{artifact_uri}/model.pkl"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"parent run {parent_run_id}")
        self.client.log_artifact(parent_run_id, local_path, artifact_uri)
        return

    def load_model(self, parent_run_id: str, run_name: str):
        """Loads a model from MLflow. Need to seperate path and file because mlflow weird"""
        remote_path = f"{self.experiment_id}/{parent_run_id}/artifacts/models/{run_name}"
        local_path = f"{TEMP_DIR}/{remote_path}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        remote_file = f"mlflow-artifacts:/{remote_path}/model.pkl"
        local_file = f"{local_path}/model.pkl"
        mlflow.artifacts.download_artifacts(
            artifact_uri=remote_file, 
            dst_path=local_path
        )
        with open(local_file, 'rb') as f:
            model = pickle.load(f)
            return model

class Repository:
    """
    Facade for MLflow tracking and artifact storage.
    Separate it for better organization and maintainability.
    This class only and should only forwarding the request to the Tracker or Artifact classes.
    Combines metrics tracking and artifact storage.
    """

    def __init__(self, tracker_path: str, experiment_name: str):
        self.metrics_tracker = Tracker(tracker_path, experiment_name)
        self.artifact_repo = Artifact(tracker_path, experiment_name)

    def start(self):
        self.active_run = mlflow.start_run(run_name=generate_random_string(6))
        return self.active_run.info.run_id

    def stop(self):
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def start_nested_run(self, run_name=None):
        mlflow.start_run(run_name=run_name, nested=True)

    def end_nested_run(self):
        mlflow.end_run()

    def end_run(self):
        self.metrics_tracker.end_run()

    def get_parent_run_id(self):
        try:
            active_run = mlflow.active_run()
            if active_run and "mlflow.parentRunId" in active_run.data.tags:
                return active_run.data.tags["mlflow.parentRunId"]
            else:
                return active_run.info.run_id
        except Exception:
            return None

    def log_metric(self, key: str, value: float):
        self.metrics_tracker.log_metric(key, value)
        return self

    def log_param(self, key: str, value: any):
        self.metrics_tracker.log_param(key, value)
        return self

    def set_tag(self, key: str, value: any):
        self.metrics_tracker.set_tag(key, value)
        return self

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str = None):
        self.artifact_repo.log_artifact(run_id, local_path, artifact_path)
        return self

    def save_model(self, model, parent_run_id: str, run_name: str):
        self.artifact_repo.save_model(model, parent_run_id, run_name)
        return self

    def save_model_manifest(self, manifest, parent_run_id: str):
        self.artifact_repo.save_model_manifest(manifest, parent_run_id)
        return self

    def load_model_manifest(self, parent_run_id: str):
        return self.artifact_repo.load_model_manifest(parent_run_id)

    def get_artifact_uri(self, run_id: str, artifact_path: str = None):
        return self.artifact_repo.get_artifact_uri(run_id, artifact_path)

    def get_active_run_id(self):
        return self.metrics_tracker.get_active_run_id()

    def load_model(self, parent_run_id: str, run_name: str):
        return self.artifact_repo.load_model(parent_run_id, run_name)

    def start_run(self, run_name=None, nested=False):
        return self.metrics_tracker.start_run(run_name, nested)

    def save_transformation(self, func, parent_run_id: str, column: str, transformation_name: str):
        return self.artifact_repo.save_transformation(func, parent_run_id, column, transformation_name)

    def load_transformation(self, func, parent_run_id: str, column: str, transformation_name: str):
        return self.artifact_repo.load_transformation(func, parent_run_id, column, transformation_name)

    def find_child_runs(self, order_by: str, filt: dict):
        parent_run_id = self.get_parent_run_id()
        return self.metrics_tracker.find_child_runs(parent_run_id, order_by, filt)

    def set_tag_run(self, run_id: str, key: str, value: any):
        self.metrics_tracker.set_tag_run(run_id, key, value)
        return self