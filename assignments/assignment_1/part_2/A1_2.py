from typing import List


def get_confusion_matrix(
    actual: List[int], predicted: List[int]
) -> List[List[int]]:
    """Computes confusion matrix from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        List of two lists of length 2 each, representing the confusion matrix.
    """
    # [[tn, fp], [fn, tp]] - task description
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0
    
    assert len(actual) == len(predicted), "actual and predicted must be of same length"
    
    for i in range(len(actual)):
        if actual[i] == 0 and predicted[i] == 0:
            tn += 1
        elif actual[i] == 0 and predicted[i] == 1:
            fp += 1
        elif actual[i] == 1 and predicted[i] == 0:
            fn += 1
        elif actual[i] == 1 and predicted[i] == 1:
            tp += 1
        else:
            raise ValueError("actual and predicted must be 0 or 1")

    return [[tn, fp], [fn, tp]]


def accuracy(actual: List[int], predicted: List[int]) -> float:
    """Computes the accuracy from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Accuracy as a float.
    """
    confusion_matrix = get_confusion_matrix(actual, predicted)
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]

    return (tp + tn) / (tp + tn + fp + fn)


def precision(actual: List[int], predicted: List[int]) -> float:
    """Computes the precision from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Precision as a float.
    """
    confusion_matrix = get_confusion_matrix(actual, predicted)
    fp = confusion_matrix[0][1]
    tp = confusion_matrix[1][1]

    return tp / (tp + fp)


def recall(actual: List[int], predicted: List[int]) -> float:
    """Computes the recall from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Recall as a float.
    """
    confusion_matrix = get_confusion_matrix(actual, predicted)
    fn = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]

    return tp / (tp + fn)


def f1(actual: List[int], predicted: List[int]) -> float:
    """Computes the F1-score from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of harmonic mean of precision and recall.
    """
    # inefficient, but who cares
    prec = precision(actual, predicted)
    rec = recall(actual, predicted)
    return 2 * (prec * rec) / (prec + rec)


def false_positive_rate(actual: List[int], predicted: List[int]) -> float:
    """Computes the false positive rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as positive divided
            by number of actually negative instances.
    """
    confusion_matrix = get_confusion_matrix(actual, predicted)
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]

    return fp / (fp + tn)


def false_negative_rate(actual: List[int], predicted: List[int]) -> float:
    """Computes the false negative rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as negative divided
            by number of actually positive instances.
    """
    confusion_matrix = get_confusion_matrix(actual, predicted)
    fn = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]

    return fn / (fn + tp)

