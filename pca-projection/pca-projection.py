import numpy as np

def pca_projection(X, k):
    X = np.array(X)
    
   
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
   
    C = np.cov(X_centered, rowvar=False)
    
    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Step 4: Sort eigenvectors by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Select top-k eigenvectors
    W = eigenvectors[:, :k]
    
    X_proj = np.dot(X_centered, W)
    
    return X_proj
    