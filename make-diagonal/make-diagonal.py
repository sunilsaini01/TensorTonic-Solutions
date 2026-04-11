import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """

    if v is None or len(v) == 0:
        return "Error: Input vector is empty"

    n = len(v)

    matrix = []

    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(v[i])   
            else:
                row.append(0)
        matrix.append(row)

    return np.array(matrix)