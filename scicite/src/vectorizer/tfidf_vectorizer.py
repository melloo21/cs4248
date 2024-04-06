from typing import Collection

from gensim import corpora
from gensim.matutils import corpus2dense
from gensim.models import TfidfModel

from src.schema.tokenized_data import TokenizedData
from src.schema.vectorized_data import VectorizedData
from src.vectorizer.abstract_vectorizer import AbstractVectorizer


class TfidfVectorizer(AbstractVectorizer):
    def __init__(self, smartirs: str = 'nfc'):
        self.dictionary = None
        self.model = None
        self.smartirs = smartirs

    def _generate_corpus(self, tokens: Collection[Collection[str]]) -> list[tuple]:
        return [self.dictionary.doc2bow(text) for text in tokens]

    def fit(self, documents: TokenizedData):
        self.dictionary = corpora.Dictionary(documents.tokens)
        corpus = self._generate_corpus(documents.tokens)
        self.model = TfidfModel(corpus, smartirs=self.smartirs)

    def transform(self, documents: TokenizedData) -> VectorizedData:
        corpus = self._generate_corpus(documents.tokens)
        vectors = corpus2dense(
            self.model[corpus],
            num_terms=len(self.dictionary),
            num_docs=len(documents.tokens),
        ).transpose()
        return VectorizedData(vectors=vectors, id=documents.id, labels=documents.labels)

    def fit_transform(self, documents: TokenizedData) -> VectorizedData:
        self.dictionary = corpora.Dictionary(documents.tokens)
        corpus = self._generate_corpus(documents.tokens)
        self.model = TfidfModel(corpus)
        vectors = corpus2dense(
            self.model[corpus],
            num_terms=len(self.dictionary),
            num_docs=len(documents.tokens),
        ).transpose()
        return VectorizedData(vectors=vectors, id=documents.id, labels=documents.labels)
