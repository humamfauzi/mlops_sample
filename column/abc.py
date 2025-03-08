from abc import ABC, abstractmethod


class TabularColumn(ABC):
    @abstractmethod
    def primary_id(cls):
        pass

    @abstractmethod
    def target(cls):
        pass

    @abstractmethod
    def categorical(cls):
        pass

    @abstractmethod
    def numerical(cls):
        pass

    @abstractmethod
    def feature(cls, current_column):
        pass
