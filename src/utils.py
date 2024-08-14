import numpy as np

def percentage_within_tolerance(actual, predictions, tolerance):
    actual = np.array(actual)
    predictions = np.array(predictions)
    difference = np.abs(actual - predictions)
    acceptable_difference = tolerance * actual
    accurate_predictions = np.where(difference <= acceptable_difference, 1, 0)
    accuracy_percentage = np.mean(accurate_predictions) * 100
    return accuracy_percentage
