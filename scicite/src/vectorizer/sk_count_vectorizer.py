from typing import Collection

from sklearn.feature_extraction.text import CountVectorizer

from src.schema.tokenized_data import TokenizedData
from src.schema.vectorized_data import VectorizedData
from src.vectorizer.abstract_vectorizer import AbstractVectorizer


class SkCountVectorizer(AbstractVectorizer):
    def __init__(
        self,
        ngram_range=(1, 2),
        token_pattern: str = r'(?u)\b[a-zA-Z0-9\-][a-zA-Z0-9\-]+\b',
        binary: bool = False,
    ):
        self.model = CountVectorizer(
            ngram_range=ngram_range, token_pattern=token_pattern, binary=binary
        )

    def _convert_corpus_format(self, corpus: Collection[Collection[str]]) -> list[str]:
        return [' '.join(document) for document in corpus]

    def fit(self, documents: TokenizedData):
        formatted_tokens = self._convert_corpus_format(documents.tokens)
        self.model.fit(formatted_tokens)

    def transform(self, documents: TokenizedData) -> VectorizedData:
        formatted_tokens = self._convert_corpus_format(documents.tokens)
        vectors = self.model.transform(formatted_tokens)
        return VectorizedData(vectors=vectors, id=documents.id, labels=documents.labels)

    def fit_transform(self, documents: TokenizedData) -> VectorizedData:
        formatted_tokens = self._convert_corpus_format(documents.tokens)
        self.model.fit(formatted_tokens)
        vectors = self.model.transform(formatted_tokens)
        return VectorizedData(vectors=vectors, id=documents.id, labels=documents.labels)
