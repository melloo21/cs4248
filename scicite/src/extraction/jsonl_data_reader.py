import json
from pathlib import Path

from src.schema.data_instance import DataInstance
from src.schema.documents import Documents
from src.utils.path_getter import PathGetter


class JsonlDataReader:
    def __init__(self, file_path: Path = None, file_name: str = None):
        if file_name:
            self._path = PathGetter.get_data_directory() / file_name
        else:
            self._path = file_path or PathGetter.get_data_directory() / 'dev.jsonl'

    def read_jsonl(self) -> list[dict]:
        with open(self._path, 'r', encoding='utf8') as f:
            data = [json.loads(file_line) for file_line in f]
        return data

    def read(self) -> Documents:
        data = self.read_jsonl()

        raw_instances = [
            DataInstance(
                row['string'], row['label'], row['id'], row['citeStart'], row['citeEnd']
            )
            for row in data
        ]
        return Documents.from_data_instance(raw_instances)
