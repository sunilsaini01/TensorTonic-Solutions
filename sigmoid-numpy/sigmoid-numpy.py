import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.array(x)
    e_power = np.exp(-x)
    result = 1/(1+e_power)
    return result
    
    pass