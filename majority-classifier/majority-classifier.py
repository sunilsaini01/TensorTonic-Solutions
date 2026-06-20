import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Find unique classes
    classes, counts = np.unique(y_train, return_counts=True)

    # majority class
    majority_class = classes[np.argmax(counts)]

    # Return array of predictions
    return np.full(len(X_test), majority_class, dtype=int)
    pass