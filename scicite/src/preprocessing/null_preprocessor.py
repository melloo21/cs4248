from src.extraction.data_instance import DataInstance
from src.preprocessing.abstract_preprocessor import AbstractPreprocessor


class NullPreprocessor(AbstractPreprocessor):
    def preprocess(self, data_instances: list[DataInstance]) -> list[DataInstance]:
        return data_instances

