import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float) 
    
    p_hat = np.clip(y_pred, eps, 1 - eps)
    loss = -((y_true * np.log(p_hat)) + ((1 - y_true) * np.log(1 - p_hat)))

    return loss.tolist()
    

    
    pass