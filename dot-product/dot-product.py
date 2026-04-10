import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    if len(x)!= len(y):         # for dot product the length of array must be equal
        raise ValueError("vector always must be in same length ")
    x = np.asarray(x)
    y = np.asarray(y)

    result = np.sum(x * y)
    return float(result)
    
    pass