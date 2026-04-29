import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.asarray(y_true, dtype = int)
    y_pred = np.asarray(y_pred, dtype = float)
    loss = -np.log(y_pred[np.arange(len(y_true)), y_true])
    result = np.mean(loss)
    return (result)
    
    
    pass