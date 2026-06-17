import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.asarray(v, dtype = float)
    if v.ndim == 1:
        norm = np.sqrt(np.sum(v**2))

        if v.ndim == 0:
            return v
            
        return v/norm

    if v.ndim == 2:
        norms = np.sqrt(np.sum(v**2, axis = 1, keepdims = True))

        norms[norms == 0] = 1
        return v / norms
    pass