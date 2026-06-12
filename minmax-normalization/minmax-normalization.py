import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    X = np.asarray(X, dtype=float)

    p = np.min(X, axis=axis, keepdims=True)        # p = min value
    q = np.max(X, axis=axis, keepdims=True)        # q = max avlue

    denominator = np.maximum(q - p, eps)

    return (X - p) / denominator