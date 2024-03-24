from abc import ABC, abstractmethod

from src.schema.tokenized_data import TokenizedData
from src.schema.vectorized_data import VectorizedData


class AbstractVectorizer(ABC):
    @abstractmethod
    def fit(self, tokens: TokenizedData):
        pass

    @abstractmethod
    def transform(self, tokens: TokenizedData) -> VectorizedData:
        pass
