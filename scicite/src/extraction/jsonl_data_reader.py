import json
from pathlib import Path

from src.extraction.data_instance import DataInstance
from src.utils.path_getter import PathGetter


class JsonlDataReader:
    def __init__(self, file_path: Path = None):
        self._path = file_path or PathGetter.get_data_directory() / 'dev.jsonl'

    def read(self) -> list[DataInstance]:
        with open(self._path, 'r', encoding='utf8') as f:
            data = [json.loads(file_line) for file_line in f]
        data_instances = [
            DataInstance(
                row['string'], row['label'], row['id'], row['citeStart'], row['citeEnd']
            )
            for row in data
        ]
        return data_instances
