import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.asarray(x, dtype=float)

    mean = float(np.mean(x))

    median = float(np.median(x))

    freq = Counter(x)
    max_count = max(freq.values())
    value = []

    for k, v in freq.items():
        if v == max_count:
            value.append(k)

    mode = float(min(value))

    return (mean, median, mode)
    pass