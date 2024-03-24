from abc import ABC, abstractmethod

from src.schema.data_instance import DataInstance
from src.schema.tokenized_data import TokenizedData


class AbstractTokenizer(ABC):
    @abstractmethod
    def tokenize(self, data_instances: list[DataInstance]) -> TokenizedData:
        pass
# TODO: Add fit and tokenize type of schema
