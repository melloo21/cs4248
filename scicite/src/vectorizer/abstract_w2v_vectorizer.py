from typing import Collection

import numpy as np

from src.schema.tokenized_data import TokenizedData
from src.schema.vectorized_data import VectorizedData
from src.vectorizer.abstract_vectorizer import AbstractVectorizer


class AbstractW2vVectorizer(AbstractVectorizer):
    def __init__(self):
        self.model = None

    def _get_cached_model(self):
        if hasattr(self.__class__, '_model'):
            return self.__class__._model
        return None

    def _cache_model(self, model):
        self.__class__._model = model


    def vectorize(self, sentence: Collection[str]) -> np.array:
        words_vecs = [self.model[word] for word in sentence if word in self.model]
        if len(words_vecs) == 0:
            return np.zeros(self.model.vector_size)
        words_vecs = np.array(words_vecs)
        return words_vecs.mean(axis=0)

    def transform(self, documents: TokenizedData) -> VectorizedData:
        tokens = documents.tokens
        vectors = np.array([self.vectorize(sentence) for sentence in tokens])
        return VectorizedData(vectors=vectors, id=documents.id, labels=documents.labels)

    def fit_transform(self, documents: TokenizedData) -> VectorizedData:
        self.fit(documents)
        return self.transform(documents)
