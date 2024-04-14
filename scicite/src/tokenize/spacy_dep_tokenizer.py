import spacy

from src.schema.documents import Documents
from src.schema.tokenized_data import TokenizedData
from src.tokenize.abstract_tokenizer import AbstractTokenizer


class SpacyDepTokenizer(AbstractTokenizer):

    def tokenize(self, document: Documents) -> TokenizedData:
        nlp = spacy.load('en_core_web_sm', disable=[])
        tokenized_docs = document.texts
        tokenized_docs = [
            [token.dep_ for token in doc]
            for doc in nlp.pipe(tokenized_docs)
        ]

        return TokenizedData(tokenized_docs, document.id, document.labels)
