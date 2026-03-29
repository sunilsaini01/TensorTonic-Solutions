def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    import numpy as np 
    import math

    list = []
    for val in x :
        if val > 0:
            list.append(val)
        else:
            list.append(alpha * (math.exp(val) - 1))

    return list 
        
 