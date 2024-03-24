import gensim.downloader as api

from src.schema.tokenized_data import TokenizedData
from src.vectorizer.abstract_w2v_vectorizer import AbstractW2vVectorizer


class FastTextW2vVectorizer(AbstractW2vVectorizer):
    def __init__(self):
        super().__init__()

    def fit(self, documents: TokenizedData):
        self.model = api.load('fasttext-wiki-news-subwords-300')
