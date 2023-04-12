import numpy as np

from mlutils.encoding.utils import encode_columns_to_labels, encode_train_test_to_labels


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


def test_train_test_encodes_correctly():
    # given
    train = np.array([
        ['a', 'b'],
        ['c', 'a'],
    ])
    test = np.array([
        ['c', 'b'],
        ['a', 'e'],
    ])

    # when
    train_encoded, test_encoded = encode_train_test_to_labels(train, test)

    # then
    assert train_encoded[0, 0] == test_encoded[1, 0]
    assert train_encoded[1, 0] == test_encoded[0, 0]
    assert train_encoded[0, 1] == test_encoded[0, 1]
    assert train_encoded[0, 1] != test_encoded[1, 1]
    assert train_encoded[1, 1] != test_encoded[1, 1]
