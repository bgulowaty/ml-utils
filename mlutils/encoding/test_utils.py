import numpy as np

from mlutils.encoding.utils import encode_columns_to_labels


def test_encodes_correctly():
    # given
    some_array = np.array([
        ['a', 'b'],
        ['c', 'b'],
    ])

    # when
    output = encode_columns_to_labels(some_array)

    # then
    assert (output == [[
        [0, 0],
        [1, 0]
    ]]).all()
