from abc import ABC
from abc import abstractmethod

class Manifest(ABC):
    @abstractmethod
    def get_model_manifest(self, id: str): 
        """
        Retrieve the model manifest associated with the specified ID.
        :param id: The unique identifier for the model.
        :return: The model manifest as a dictionary.
        """

    @abstractmethod
    def get_transformation_manifest(self, id: str):
        """
        Retrieve the transformation manifest associated with the specified ID.
        :param id: The unique identifier for the transformation.
        :return: The transformation manifest as a dictionary.
        """

    @abstractmethod
    def get_metadata_manifest(self, id: str):
        """
        Retrieve the metadata manifest associated with the specified ID.
        :param id: The unique identifier for the metadata.
        :return: The metadata manifest as a dictionary.
        """

    @abstractmethod
    def save_model_manifest(self, id: str, manifest: dict):
        """
        Save the model manifest associated with the specified ID.
        :param id: The unique identifier for the model.
        :param manifest: The model manifest to be saved (dictionary).
        """

    @abstractmethod
    def save_transformation_manifest(self, id: str, manifest: dict):
        """
        Save the transformation manifest associated with the specified ID.
        :param id: The unique identifier for the transformation.
        :param manifest: The transformation manifest to be saved (dictionary).
        """

    @abstractmethod
    def save_metadata_manifest(self, id: str, manifest: dict):
        """
        Save the metadata manifest associated with the specified ID.
        :param id: The unique identifier for the metadata.
        :param manifest: The metadata manifest to be saved (dictionary).
        """