import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_pred = np.asarray(y_pred, dtype = float)
    y_true = np.asarray(y_true, dtype = float)
    exp = y_true - y_pred

    result = np.where(np.abs(exp) <= delta , (1/2)*exp**2 , delta*(np.abs(exp) - (1/2)*delta))

    return np.mean(result)
    pass