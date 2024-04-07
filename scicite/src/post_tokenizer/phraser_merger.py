from gensim.models import Phrases
from gensim.models.phrases import Phraser

from src.post_tokenizer.abstract_post_tokenizer import AbstractPostTokenizer
from src.schema.tokenized_data import TokenizedData


class SinglePhraserMerger(AbstractPostTokenizer):
    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
        self.model = None

    def fit(self, documents: TokenizedData):
        phrases = Phrases(documents.tokens, threshold=self.threshold)
        self.model = Phraser(phrases)

    def transform(self, documents: TokenizedData) -> TokenizedData:
        transformed = list(self.model[documents.tokens])
        transformed = [list(set(original+bigram)) for original, bigram in zip(documents.tokens, transformed)]
        return TokenizedData(transformed, documents.id, documents.labels)


class PhraserMerger(AbstractPostTokenizer):
    def __init__(self, num_gram: int = 1, threshold: float = 2.0):
        self.num_gram = num_gram
        self.threshold = threshold
        self.models = [SinglePhraserMerger(threshold=threshold) for _ in range(num_gram)]

    def fit(self, documents: TokenizedData):
        self.fit_transform(documents)

    def transform(self, documents: TokenizedData) -> TokenizedData:
        modified_documents = documents
        for model in self.models:
            modified_documents = model.transform(modified_documents)
        return modified_documents

    def fit_transform(self, documents: TokenizedData) -> TokenizedData:
        modified_documents = documents
        for model in self.models:
            modified_documents = model.fit_transform(modified_documents)
        return modified_documents
