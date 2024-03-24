from src.preprocessing.abstract_preprocessor import AbstractPreprocessor
from src.schema.documents import Documents


class NullPreprocessor(AbstractPreprocessor):
    def preprocess(self, document: Documents) -> Documents:
        return document
