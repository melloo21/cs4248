from dataclasses import dataclass, field
from typing import Collection

from src.schema.data_instance import DataInstance
from src.schema.labels import Labels, LabelIndices
from src.utils.label_mapping import map_label_string_to_indices


@dataclass
class Documents:
    raw_instances: Collection[DataInstance]
    texts: Collection[str]
    id: Collection[str]
    labels: Collection[Labels]
    label_indices: Collection[LabelIndices] = field(init=False)

    def __post_init__(self):
        assert len(self.texts) == len(self.id) == len(self.labels), (
            f'Inputs do not have the same data instance counts: '
            f'{len(self.texts)}, {len(self.id)}, {len(self.labels)}'
        )
        self.label_indices = map_label_string_to_indices(self.labels)
