from abc import ABC, abstractmethod

from src.schema.tokenized_data import TokenizedData
from src.schema.vectorized_data import VectorizedData


class AbstractVectorizer(ABC):
    @abstractmethod
    def vectorize(self, tokens: TokenizedData) -> VectorizedData:
        pass
