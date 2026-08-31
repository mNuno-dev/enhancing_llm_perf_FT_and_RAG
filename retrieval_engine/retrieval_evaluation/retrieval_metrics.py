def precision(tp, tn, fp, fn):
    """
    Calculate the precision of a binary classification model.

    Precision is the ratio of true positives (TP) to the sum of true positives
    and false positives (FP), which indicates the accuracy of positive predictions.

    Args:
        tp (int): True positives.
        tn (int): True negatives (not used in precision calculation).
        fp (int): False positives.
        fn (int): False negatives (not used in precision calculation).

    Returns:
        float: The precision value, or 0.0 if the denominator is zero.
    
    Raises:
        ValueError: If any input is negative.
    """
    if tp < 0 or tn < 0 or fp < 0 or fn < 0:
        raise ValueError("All input values must be non-negative.")
    
    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall(tp, tn, fp, fn):
    """
    Calculate the recall (sensitivity) of a binary classification model.

    Recall is the ratio of true positives (TP) to the sum of true positives (TP)
    and false negatives (FN), measuring the ability to detect positive instances.

    Args:
        tp (int): True positives.
        tn (int): True negatives (not used in recall calculation).
        fp (int): False positives (not used in recall calculation).
        fn (int): False negatives.

    Returns:
        float: The recall value, or 0.0 if the denominator is zero.
    
    Raises:
        ValueError: If any input is negative.
    """
    if tp < 0 or tn < 0 or fp < 0 or fn < 0:
        raise ValueError("All input values must be non-negative.")

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def MRR():
    ...

def MAP():
    ...

# Normalized Discounted Cumulative Gain (NDCG)
# Reference: https://weaviate.io/blog/retrieval-evaluation-metrics
# This is the default metric used in the MTEB Leaderboard for the Retrieval category.
def NDCG():
    ...

def hit_rate():
    ...
