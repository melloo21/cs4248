from abc import ABC, abstractmethod

from src.schema.documents import Documents


class AbstractPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, document: Documents) -> Documents:
        pass
