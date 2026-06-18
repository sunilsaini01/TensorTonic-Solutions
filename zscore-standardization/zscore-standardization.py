import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    X = np.asarray(X, dtype=float)

    p = np.mean(X, axis=axis, keepdims=True)   # p = mean
    q = np.std(X, axis=axis, keepdims=True)    # q = std

    Y = (X - p) / (q + eps)

    return Y
    pass