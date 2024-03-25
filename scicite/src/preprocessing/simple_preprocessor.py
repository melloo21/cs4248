import pandas as pd

from src.preprocessing.abstract_preprocessor import AbstractPreprocessor
from src.preprocessing.train_duplicate_remover import TrainDuplicateRemover
from src.schema.data_instance import DataInstance
from src.schema.documents import Documents


class SimplePreprocessor(AbstractPreprocessor):
    def __init__(self, remove_citations: bool = True, remove_duplicates: bool = True):
        self._remove_citations = remove_citations
        self._remove_duplicates = remove_duplicates

    @staticmethod
    def remove_citations(data_instance: DataInstance) -> DataInstance:
        cleaned_text = data_instance.string
        if pd.notnull(data_instance.citeStart) and pd.notnull(data_instance.citeEnd):
            cleaned_text = (
                cleaned_text[: int(data_instance.citeStart)]
                + cleaned_text[int(data_instance.citeEnd) :]
            )
        return DataInstance(
            string=cleaned_text,
            label=data_instance.label,
            id=data_instance.id,
            citeStart=data_instance.citeStart,
            citeEnd=data_instance.citeEnd,
        )

    def preprocess(self, document: Documents) -> Documents:
        preprocessed_instances = document
        if self._remove_citations:
            preprocessed_instances = list(
                map(self.remove_citations, document.raw_instances)
            )
        if self._remove_duplicates:
            preprocessed_instances = TrainDuplicateRemover().remove_if_train(
                preprocessed_instances
            )
        return Documents.from_data_instance(preprocessed_instances)
