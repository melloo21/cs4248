from typing import Iterable

from src.schema.labels import LabelIndices, Labels


def map_label_string_to_indices(labels: Iterable[Labels]) -> list[LabelIndices]:
    mapping = dict(zip(['background', 'method', 'result'], [0, 1, 2]))
    return [mapping[label] for label in labels]


def map_label_number_to_string(label_numbers: Iterable[LabelIndices]) -> list[Labels]:
    mapping = dict(zip([0, 1, 2], ['background', 'method', 'result']))
    return [mapping[label] for label in label_numbers]

