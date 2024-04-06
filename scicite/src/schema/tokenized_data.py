from dataclasses import dataclass
from typing import Collection

from src.schema.labels import Labels


@dataclass
class TokenizedData:
    tokens: Collection[Collection[str]]
    id: Collection[str]
    labels: Collection[Labels]

    def __post_init__(self):
        assert len(self.tokens) == len(self.id) == len(self.labels), (
            f'Inputs do not have the same data instance counts: '
            f'{len(self.tokens)}, {len(self.id)}, {len(self.labels)}'
        )
