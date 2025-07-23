""" """
from typing import Tuple

import numpy as np


def get_confidence_scores(
        y: np.array,
        y_pred: np.array,
        threshold: float = 0.5,
) -> Tuple[float, float]:
    """ Calculate the likelihood for the false negative: get the mean value of
    the prediction for the false-positive and false-negatives
    
    Args:
        y (np.array): True labels.
        y_pred (np.array): Predicted probabilities.
        threshold (float): Threshold to classify predictions as positive or negative.
        
    Returns:
        Tuple[float, float]: Mean prediction values for false positives and false negatives.
    """
    # Ensure y and y_pred are numpy arrays
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    # Get the indices of the false positives and false negatives
    false_positives = (y == 0) & ((y_pred > threshold).astype(int) == 1)
    false_negatives = (y == 1) & ((y_pred > threshold).astype(int) == 0)

    # Get the mean value of the predictions for the false positives and false negatives
    false_positives_mean = y_pred[false_positives].mean()
    false_negatives_mean = y_pred[false_negatives].mean()

    return false_positives_mean, false_negatives_mean