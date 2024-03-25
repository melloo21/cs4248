from abc import ABC, abstractmethod

from src.schema.documents import Documents
from src.schema.tokenized_data import TokenizedData


class AbstractTokenizer(ABC):
    def fit(self, document: Documents):
        pass

    @abstractmethod
    def tokenize(self, document: Documents) -> TokenizedData:
        pass
