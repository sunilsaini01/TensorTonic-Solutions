import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    XT = X.T
    XTX = XT @ X
    XTy = XT @ y

    XTX_inv = np.linalg.inv(XTX)

    w = XTX_inv @ XTy

    return w
    pass