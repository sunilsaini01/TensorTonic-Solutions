import numpy as np

def streaming_minmax_init(D):
    """
    Initialize streaming min-max scaler state.
    
    Parameters
    """
    return {
        "min": np.full(D, np.inf),     # D = no. of feature/column
        "max": np.full(D, -np.inf)
    }


def streaming_minmax_update(state, X_batch, eps=1e-12):
    """
    Update running min/max using a new batch and
    return the normalized batch.
    """
    
    X_batch = np.asarray(X_batch, dtype=float)
    
    # Batch statistics
    batch_min = np.min(X_batch, axis=0)
    batch_max = np.max(X_batch, axis=0)
    
    # Update running statistics
    state["min"] = np.minimum(state["min"], batch_min)
    state["max"] = np.maximum(state["max"], batch_max)
    
    # Compute range safely
    ranges = np.maximum(state["max"] - state["min"], eps)
    
    # Min-Max Scaling
    result = (X_batch - state["min"]) / ranges
    
    return result