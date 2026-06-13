import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    g = np.asarray(grad, dtype=float)

    # Look ahead position
    w_look = w - momentum * v

    # Update velocity
    v = momentum * v + lr * g

    # Update weights
    w = w - v

    return w, v
    pass