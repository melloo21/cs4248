from src.post_tokenizer.abstract_post_tokenizer import AbstractPostTokenizer
from src.schema.tokenized_data import TokenizedData


class NullPostTokenizer(AbstractPostTokenizer):
    def transform(self, documents: TokenizedData) -> TokenizedData:
        return documents
