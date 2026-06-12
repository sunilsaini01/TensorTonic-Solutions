import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    G = np.asarray(G, dtype = float)
    w = np.asarray(w, dtype = float)
    g = np.asarray(g, dtype = float)
    
    # Accumulate Squared Gradients
    G_new = G + g**2

    # parameter Update
    w_new = w - (lr/(np.sqrt(G_new + eps))) * g

    return w_new, G_new
    pass