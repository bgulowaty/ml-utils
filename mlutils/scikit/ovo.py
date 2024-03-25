import numpy as np


def ovo(function, x, y):
    complexities = []

    for label in np.unique(y):
        ovo_y = y.copy()
        negative_idx = ovo_y != label
        positive_idx = ovo_y == label

        ovo_y[negative_idx] = 0
        ovo_y[positive_idx] = 1

        if len(np.unique(ovo_y)) != 1:
            complexity = function(x, ovo_y)
            complexities.append(complexity)
        else:
            complexities.append(0)

    return complexities
