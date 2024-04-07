from abc import ABC, abstractmethod

from src.schema.tokenized_data import TokenizedData


class AbstractPostTokenizer(ABC):
    def fit(self, documents: TokenizedData):
        pass

    @abstractmethod
    def transform(self, documents: TokenizedData) -> TokenizedData:
        pass

    def fit_transform(self, documents: TokenizedData) -> TokenizedData:
        self.fit(documents)
        return self.transform(documents)
