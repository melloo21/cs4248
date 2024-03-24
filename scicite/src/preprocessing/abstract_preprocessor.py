from abc import ABC, abstractmethod

from src.extraction.data_instance import DataInstance


class AbstractPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, data_instances: list[DataInstance]) -> list[DataInstance]:
        pass
