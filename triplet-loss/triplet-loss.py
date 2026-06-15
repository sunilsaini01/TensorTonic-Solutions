import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    anchor = np.asarray(anchor)
    positive = np.asarray(positive)
    negative = np.asarray(negative)

    # Single triplet
    if anchor.ndim == 1:
        d_ap = np.sum((anchor - positive) ** 2)
        d_an = np.sum((anchor - negative) ** 2)

        return float(max(0, d_ap - d_an + margin))

    # Batch of triplets
    d_ap = np.sum((anchor - positive) ** 2, axis=1)
    d_an = np.sum((anchor - negative) ** 2, axis=1)

    loss = np.maximum(0, d_ap - d_an + margin)

    # Return average loss
    return float(np.mean(loss))
    pass