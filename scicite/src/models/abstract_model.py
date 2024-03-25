from abc import ABC, abstractmethod
from typing import Iterable

from sklearn.metrics import f1_score

from src.schema.labels import LabelIndices
from src.schema.vectorized_data import VectorizedData


class AbstractModel(ABC):
    @abstractmethod
    def fit(self, data_vectors: VectorizedData):
        pass

    @abstractmethod
    def predict(self, data_vectors: VectorizedData) -> Iterable[LabelIndices]:
        pass

    def fit_predict(self, data_vectors: VectorizedData) -> Iterable[LabelIndices]:
        self.fit(data_vectors)
        return self.predict(data_vectors)

    def score(self, y: Iterable[LabelIndices], data_vectors: VectorizedData) -> float:
        y_predict = self.predict(data_vectors)
        return f1_score(y, y_predict, average='macro')
