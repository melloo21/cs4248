import pandas as pd

from src.utils.path_getter import PathGetter


class TrainDuplicateRemover:
    train_data_size = 8243
    duplicate_set = None

    def _init_duplicate_row_set(self) -> set[int]:
        citation_path = PathGetter.get_data_directory() / 'duplicate_citations.csv'
        duplicate_csv = pd.read_csv(citation_path)
        duplicate_set = set(duplicate_csv['remove_row'].unique())
        return duplicate_set

    def get_duplicate_row_set(self) -> set[int]:
        if self.__class__.duplicate_set is not None:
            return self.__class__.duplicate_set
        self.__class__.duplicate_set = self._init_duplicate_row_set()
        return self.__class__.duplicate_set

    def remove(self, jsonl_data: list) -> list:
        assert len(jsonl_data) == self.train_data_size
        duplicate_set = self.get_duplicate_row_set()
        filtered_data = [
            row for idx, row in enumerate(jsonl_data) if idx not in duplicate_set
        ]
        return filtered_data

    def remove_if_train(self, jsonl_data: list) -> list:
        if len(jsonl_data) == self.train_data_size:
            return self.remove(jsonl_data)
        return jsonl_data
