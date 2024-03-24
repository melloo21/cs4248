from typing import Type

import pytest

from src.utils.label_mapping import (
    map_label_string_to_indices,
    map_label_number_to_string,
)


@pytest.mark.parametrize(
    ['labels', 'expected_indices'],
    [
        (['background', 'result', 'method'], [0, 2, 1]),
    ],
)
def test_map_label_string_to_indices(labels, expected_indices):
    assert map_label_string_to_indices(labels) == expected_indices


@pytest.mark.parametrize(
    ['labels', 'expected_error'],
    [
        (['background', 'somethingelse', 'method'], KeyError),
    ],
)
def test_map_label_string_to_indices_error(labels, expected_error: Type[Exception]):
    with pytest.raises(expected_error):
        map_label_string_to_indices(labels)

@pytest.mark.parametrize(
    ['indices', 'expected_labels'],
    [
        ([2, 1, 0], ['result', 'method', 'background']),
    ],
)
def test_map_label_number_to_string(indices, expected_labels):
    assert map_label_number_to_string(indices) == expected_labels


@pytest.mark.parametrize(
    ['indices', 'expected_error'],
    [
        ([2, 1, 3], KeyError),
    ],
)
def test_map_label_string_to_indices_error(indices, expected_error: Type[Exception]):
    with pytest.raises(expected_error):
        map_label_number_to_string(indices)
