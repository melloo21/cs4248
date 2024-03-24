from __future__ import annotations

from dataclasses import dataclass, field
from operator import attrgetter
from typing import Collection

import numpy as np
from sklearn.model_selection import train_test_split

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

    def split(
        self,
        test_size: float | int = None,
        train_size: float | int = None,
        random_state: int = None,
        shuffle: bool = True,
        stratify: np.array = None,
    ) -> tuple[Documents, Documents]:
        split_data = train_test_split(
            self.raw_instances,
            self.texts,
            self.id,
            self.labels,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        train_data = split_data[::2]
        test_data = split_data[1::2]
        return Documents(*train_data), Documents(*test_data)

    @classmethod
    def from_data_instance(cls, data_instances: Collection[DataInstance]) -> Documents:
        data = {
            'raw_instances': data_instances,
            'texts': list(map(attrgetter('string'), data_instances)),
            'id': list(map(attrgetter('id'), data_instances)),
            'labels': list(map(attrgetter('label'), data_instances)),
        }
        return Documents(**data)
