from io import BytesIO

import sentencepiece

from src.schema.documents import Documents
from src.schema.tokenized_data import TokenizedData
from src.tokenize.abstract_tokenizer import AbstractTokenizer


class SentencePieceTokenizer(AbstractTokenizer):
    def __init__(self, vocab_size: int = 10000):
        self.model = None
        self.vocab_size = vocab_size

    def fit(self, document: Documents):
        text_input = BytesIO(bytes('\n'.join(document.texts), 'utf-8'))
        tokenizer_model = BytesIO()
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=text_input,
            model_writer=tokenizer_model,
            vocab_size=self.vocab_size,
        )
        self.model = sentencepiece.SentencePieceProcessor(
            model_proto=tokenizer_model.getvalue()
        )

    def tokenize(self, document: Documents) -> TokenizedData:
        encoded_text = self.model.encode(document.texts, out_type=str)
        return TokenizedData(encoded_text, document.id, document.labels)
