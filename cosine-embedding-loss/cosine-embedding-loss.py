import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # Compute cosine similarity
    dot_product = np.dot(x1, x2)

    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)

    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return None

    cosine_similarity = dot_product / (norm1 * norm2)

    # Loss calculation
    if label == 1:
        return 1 - cosine_similarity

    elif label == -1:
        return max(0, cosine_similarity - margin)

    else:
        return None