import json
from repositories.abc import Manifest

class DummyMLflowRepository(Manifest):
    """
    A dummy implementation of the MLflowRepository for testing purposes.
    It mocks the behavior of MLflow without actually interacting with an MLflow server.
    """

    def __init__(self, experiment_name: str, base_path: str):
        """
        Initializes the DummyMLflowRepository.
        :param experiment_name: The name of the experiment (for mock purposes).
        :param base_path: The base path for saving artifacts (in memory).
        """
        self.experiment_name = experiment_name
        self.base_path = base_path
        self.manifests = {}  # Store manifests in memory

    def start(self):
        """
        Mocks starting an MLflow run.
        :return: A dummy run ID.
        """
        self.run_id = "dummy_run_id"
        return self.run_id

    def stop(self):
        """
        Mocks stopping an MLflow run.
        """
        pass

    def get_parent_run_id(self):
        """
        Mocks retrieving the active run ID.
        :return: A dummy run ID.
        """
        return "dummy_parent_run_id"

    def get_experiment_id(self):
        """
        Mocks retrieving the experiment ID.
        :return: A dummy experiment ID.
        """
        return "dummy_experiment_id"

    def get_model_manifest(self, id: str):
        """
        Retrieves a model manifest from the in-memory store.
        :param id: The unique identifier for the model.
        :return: The model manifest as a dictionary, or None if not found.
        """
        return self.manifests.get(f"model_manifest_{id}", None)

    def get_transformation_manifest(self, id: str):
        """
        Retrieves a transformation manifest from the in-memory store.
        :param id: The unique identifier for the transformation.
        :return: The transformation manifest as a dictionary, or None if not found.
        """
        return self.manifests.get(f"transformation_manifest_{id}", None)

    def get_metadata_manifest(self, id: str):
        """
        Retrieves a metadata manifest from the in-memory store.
        :param id: The unique identifier for the metadata.
        :return: The metadata manifest as a dictionary, or None if not found.
        """
        return self.manifests.get(f"metadata_manifest_{id}", None)

    def save_model_manifest(self, id: str, manifest: dict):
        """
        Saves a model manifest to the in-memory store.
        :param id: The unique identifier for the model.
        :param manifest: The model manifest to be saved (dictionary).
        """
        self.manifests[f"model_manifest_{id}"] = manifest

    def save_transformation_manifest(self, parent_run_id: str, manifest: dict):
        """
        Saves a transformation manifest to the in-memory store.
        :param id: The unique identifier for the transformation.
        :param manifest: The transformation manifest to be saved (dictionary).
        """
        self.manifests[f"transformation_manifest_{id}"] = manifest

    def save_metadata_manifest(self, id: str, manifest: dict):
        """
        Saves a metadata manifest to the in-memory store.
        :param id: The unique identifier for the metadata.
        :param manifest: The metadata manifest to be saved (dictionary).
        """
        self.manifests[f"metadata_manifest_{id}"] = manifest

    def load_data(self, path: str) -> dict:
        """
        Load data from the specified path.
        :param path: The path to the data file.
        :return: The loaded data as a dictionary.
        """
        pass

    def save_data(self, path: str, data: dict):
        """
        Save data to the specified path.
        :param path: The path to save the data file.
        :param data: The data to be saved (dictionary).
        """
        pass
    
    def save_folder(self, path: str):
        """
        Save the folder to the specified path.
        :param path: The path to save the folder.
        """
        pass

    def get_parent_run_id(self):
        """
        Mocks retrieving the parent run ID.
        :return: A dummy parent run ID.
        """
        return "dummy_parent_run_id"

    def save_transformation(self, func, parent_run_id: str, column: str, transformation_name: str):
        pass