import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_columns_to_labels(numpy_array):
    if len(numpy_array.shape) != 2:
        raise Exception("Only 2d arrays are supported")

    le = LabelEncoder()

    return np.apply_along_axis(lambda col: le.fit_transform(col), 0, numpy_array)
