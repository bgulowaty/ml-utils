import numpy as np
from loguru import logger as log

def find_closest_val(arr, val):
    return np.argmin(np.abs(np.array(arr) - val))


def compute_metric_ovo(x, y, metric):
    metric_values = []
    for label in np.unique(y):
        ovo_y = y.copy()
        zero_indices = ovo_y != label
        one_indices = ovo_y == label

        ovo_y[zero_indices] = 0
        ovo_y[one_indices] = 1
        metric_value = metric(x, ovo_y)
        metric_values.append(metric_value)

    return metric_values

def list_with_repeated_elements(input_list, n_repeated):
    return [val for val in input_list for _ in range(n_repeated)]

def exclude_indices(arr, indices):
    return arr[~np.isin(np.arange(arr.shape[0]), indices)]
