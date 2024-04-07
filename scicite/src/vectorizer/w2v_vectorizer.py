from gensim.models import Word2Vec

from src.schema.tokenized_data import TokenizedData
from src.vectorizer.abstract_w2v_vectorizer import AbstractW2vVectorizer


class W2vVectorizer(AbstractW2vVectorizer):
    def __init__(
        self,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 5,
        hierarchical_sampling: int = 1,
        negative: int = 0,
        workers: int = 4,
        epochs: int = 300,
    ):
        super().__init__()
        self.model = Word2Vec(
            vector_size=vector_size,
            min_count=min_count,
            window=window,
            workers=workers,
            hs=hierarchical_sampling,
            negative=negative,
        )
        self.epochs = epochs

    def fit(self, documents: TokenizedData):
        self.model.build_vocab(documents.tokens)
        self.model.train(
            documents.tokens,
            total_examples=self.model.corpus_count,
            epochs=self.epochs,
            report_delay=1,
        )
        self.model.init_sims(replace=True)
        self.model = self.model.wv
