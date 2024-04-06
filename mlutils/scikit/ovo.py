import numpy as np


def ovo(function, x, y):
    complexities = []

    for label in np.unique(y):
        complexity = function(x, (y == label))
        complexities.append(complexity)

    return complexities
