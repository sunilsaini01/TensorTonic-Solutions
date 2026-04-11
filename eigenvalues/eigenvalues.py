import numpy as np

def calculate_eigenvalues(matrix):
    # check if matrix is valid list
    if matrix is None or not isinstance(matrix, list) or len(matrix) == 0:
        return None

    if not isinstance(matrix[0], list):
        return None

    if len(matrix[0]) == 0:
        return None

    row_length = len(matrix[0])
    for row in matrix:
        if not isinstance(row, list) or len(row) != row_length:
            return None

    x = np.array(matrix)

    if x.shape[0] != x.shape[1]:
        return None

    result = np.linalg.eigvals(x)
    result = result[np.lexsort((result.imag, result.real))]

    return result