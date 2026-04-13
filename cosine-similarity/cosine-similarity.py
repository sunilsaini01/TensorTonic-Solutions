import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.asarray(a)
    b = np.asarray(b)

    x = np.dot(a, b)    
    p = np.linalg.norm(a)
    q = np.linalg.norm(b)
    y = p * q 
    if y == 0:
        return 0
    result = float((x)/y)
    return result
    pass