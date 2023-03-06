import pytest

from mlutils.datasets.read_dataset import read_dataset

TRAIN_TEST_TUPLES = [
    ('test-resources/nursery-train-1-s2.csv', 'test-resources/nursery-train-1-s2.csv'),
    ('test-resources/breast-train-0-s1.csv', 'test-resources/breast-test-0-s1.csv'),
    ('test-resources/hepatitis-train-3-s1.csv', 'test-resources/hepatitis-test-3-s1.csv'),
    ('test-resources/abalone-train-2-s2.csv', 'test-resources/abalone-test-2-s2.csv'),
]


@pytest.mark.parametrize(
    "train_path,test_path",
    TRAIN_TEST_TUPLES
)
def test_reads_datasets(train_path, test_path):
    # expect
    read_dataset(train_path)
    read_dataset(test_path)

    # to pass
