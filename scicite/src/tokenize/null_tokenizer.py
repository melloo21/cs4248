from src.schema.documents import Documents
from src.schema.tokenized_data import TokenizedData
from src.tokenize.abstract_tokenizer import AbstractTokenizer


class NullTokenizer(AbstractTokenizer):
    def tokenize(self, document: Documents) -> TokenizedData:
        texts = [list(doc.split()) for doc in document.texts]
        return TokenizedData(texts, document.id, document.labels)
