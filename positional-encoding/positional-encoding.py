import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    pe = np.zeros((seq_len, d_model), dtype=float)

    positions = np.arange(seq_len).reshape(-1, 1)

    div_term = np.power(base, np.arange(0, d_model, 2) / d_model)

    pe[:, 0::2] = np.sin(positions / div_term)

    if d_model > 1:
        pe[:, 1::2] = np.cos(
            positions / div_term[:pe[:, 1::2].shape[1]]
        )

    return pe
    pass