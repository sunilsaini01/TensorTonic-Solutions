import numpy as np 
import math

def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    values = np.asarray(values, dtype=float)

    if min(values) == max(values):
        return [0] * len(values)
        
    w = (max(values)-min(values))/num_bins

    bin=[]
    for i in range(len(values)):
        bin.append(min(math.floor((values[i]-min(values))/w), num_bins-1))

    return bin