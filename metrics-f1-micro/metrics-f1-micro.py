def f1_micro(y_true, y_pred) -> float:
    TP = 0
    FP = 0
    FN = 0
    
    # Get all unique classes
    classes = set(y_true) | set(y_pred)
    
    for cls in classes:
        for yt, yp in zip(y_true, y_pred):
            if yt == cls and yp == cls:
                TP += 1
            elif yt != cls and yp == cls:
                FP += 1
            elif yt == cls and yp != cls:
                FN += 1
    
 
    if (2 * TP + FP + FN) == 0:
        return 0.0
    
    return (2 * TP) / (2 * TP + FP + FN)
    pass