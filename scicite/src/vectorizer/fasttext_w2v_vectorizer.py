import gensim.downloader as api

from src.schema.tokenized_data import TokenizedData
from src.vectorizer.abstract_w2v_vectorizer import AbstractW2vVectorizer


class FastTextW2vVectorizer(AbstractW2vVectorizer):
    def __init__(self):
        super().__init__()

    def fit(self, documents: TokenizedData):
        model = self._get_cached_model()
        if model:
            self.model = model
        else:
            self.model = api.load('fasttext-wiki-news-subwords-300')
            self._cache_model(self.model)
