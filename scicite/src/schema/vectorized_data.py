from dataclasses import dataclass, field
from numbers import Number
from typing import Collection

from src.schema.labels import Labels, LabelIndices
from src.utils.label_mapping import map_label_string_to_indices


@dataclass
class VectorizedData:
    vectors: Collection[Collection[Number]]
    id: Collection[str]
    labels: Collection[Labels]
    label_indices: Collection[LabelIndices] = field(init=False)

    def __post_init__(self):
        assert len(self.vectors) == len(self.id) == len(self.labels), (
            f'Inputs do not have the same data instance counts: '
            f'{len(self.vectors)}, {len(self.id)}, {len(self.labels)}'
        )
        self.label_indices = map_label_string_to_indices(self.labels)
