import numpy as np
import pytest
from box import Box

from mlutils.scikit.ovo import ovo

X_SUMMING_CUTOFF_FUNCTION = lambda x, y: np.sum(x[y == 1, :])

@pytest.mark.parametrize("dataset,expected_complexities", [
    [{"x": np.array([[1, 1], [2, 3]]), "y": np.array([1, 2])}, [2, 5]],
    [{"x": np.array([[1, 1], [2, 3]]), "y": np.array([1, 1])}, [7]],
    [{"x": np.array([[1, 1], [2, 3], [2, 2], [1, 1]]), "y": np.array([1, 2, 2, 3])}, [2, 9, 2]]
])
def test_compute_complexities_ovo(dataset, expected_complexities):
    # given
    dataset = Box(dataset)

    # when
    complexities = ovo(X_SUMMING_CUTOFF_FUNCTION, dataset.x, dataset.y)  # sum only X for given class

    # then
    assert complexities == expected_complexities
