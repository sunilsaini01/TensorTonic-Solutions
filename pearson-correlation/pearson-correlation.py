import numpy as np

def pearson_correlation(X):
    X = np.array(X, dtype=float)

    if X.ndim != 2 or X.shape[0] < 2:
        return None

    n = X.shape[0]

    # Center data
    X_centered = X - np.mean(X, axis=0)

    # Covariance matrix
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)

    # Standard deviations
    std_devs = np.std(X, axis=0, ddof=1)

    # Denominator matrix
    denom = np.outer(std_devs, std_devs)

    # Correlation matrix
    corr_matrix = cov_matrix / denom

    # Handle zero-variance features
    zero_var = (std_devs == 0)

    if np.any(zero_var):
        corr_matrix[:, zero_var] = np.nan
        corr_matrix[zero_var, :] = np.nan

    # Set diagonal only for non-zero variance features
    for i in range(len(std_devs)):
        if std_devs[i] != 0:
            corr_matrix[i, i] = 1.0

    return corr_matrix