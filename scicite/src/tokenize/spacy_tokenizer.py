import spacy
from spacy.tokens.token import Token
from spacy_cleaner.processing import replace_number_token

from src.tokenize.abstract_tokenizer import AbstractTokenizer


class SpacyTokenizer(AbstractTokenizer):

    def __init__(
            self,
            merge_nouns: bool = False,
            merge_entities: bool = False,
            # replace_punctuations: bool = True,
            remove_stopwords: bool = True,
            replace_numbers: bool = True,
            lowercase: bool = True,
            lemmatize: bool = True,
    ):
        self.merge_nouns = merge_nouns
        self.merge_entities = merge_entities
        # self.replace_punctuations = replace_punctuations
        self.remove_stopwords = remove_stopwords
        self.replace_numbers = replace_numbers
        self.lowercase = lowercase
        self.lemmatize = lemmatize

    def tokenize(self, document: list[str]) -> list[list[str]]:
        nlp = spacy.load('en_core_web_sm', disable=[])
        if self.merge_nouns:
            nlp.add_pipe('merge_noun_chunks')
        if self.merge_entities:
            nlp.add_pipe('merge_entities')
        tokenized_docs = document
        if self.remove_stopwords:
            tokenized_docs = [
                list(filter(lambda x: not x.is_stop and not x.is_punct, doc))
                for doc in nlp.pipe(tokenized_docs)
            ]
        else:
            tokenized_docs = [
                list(filter(lambda x: not x.is_punct, doc))
                for doc in nlp.pipe(tokenized_docs)
            ]

        if self.replace_numbers:
            tokenized_docs = [
                list(map(replace_number_token, doc)) for doc in tokenized_docs
            ]
        # if self.replace_punctuations:
        #     tokenized_docs = [
        #         list(map(replace_punctuation_token, doc)) for doc in tokenized_docs
        #     ]
        if self.lemmatize:
            tokenized_corpus = [
                [x.lemma_ if isinstance(x, Token) else str(x) for x in doc]
                for doc in tokenized_docs
            ]
        else:
            tokenized_corpus = [
                [str(x) for x in doc]
                for doc in tokenized_docs
            ]

        if self.lowercase:
            tokenized_corpus = [
                [token.lower() for token in doc]
                for doc in tokenized_corpus
            ]
        return tokenized_corpus
