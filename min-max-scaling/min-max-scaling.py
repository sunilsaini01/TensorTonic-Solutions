import numpy as np
def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    x = np.asarray(data, dtype=float)

    col_min = np.min(x, axis=0)
    col_max = np.max(x, axis=0)

    ranges = col_max - col_min

    ranges[ranges == 0] = 1

    scaled = (x - col_min) / ranges

    return scaled.tolist()