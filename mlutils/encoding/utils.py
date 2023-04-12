import numpy as np
from sklearn.preprocessing import LabelEncoder

from mlutils.encoding.refittable_label_encoder import RefitableLabelEncoder


def encode_columns_to_labels(numpy_array):
    if len(numpy_array.shape) != 2:
        raise Exception("Only 2d arrays are supported")

    le = LabelEncoder()

    return np.apply_along_axis(lambda col: le.fit_transform(col), 0, numpy_array)

def encode_train_test_to_labels(train_dataset, test_dataset):
    if len(train_dataset.shape) != 2 or len(test_dataset.shape) != 2:
        raise Exception("Only 2d arrays are supported")

    encoders = [RefitableLabelEncoder() for _ in range(train_dataset.shape[1])]

    encoders_iterator = iter(encoders)
    train_encoded = np.apply_along_axis(lambda col: next(encoders_iterator).fit_transform(col), 0, train_dataset)

    encoders_iterator = iter(encoders)
    test_encoded = np.apply_along_axis(lambda col: next(encoders_iterator).fit_transform(col), 0, test_dataset)

    return train_encoded, test_encoded

