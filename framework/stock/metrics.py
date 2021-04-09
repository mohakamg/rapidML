from typing import Dict


def get_regression_metrics() -> Dict:
    from framework.metrics.mean_absolute_error import MeanAbsoluteError

    REGRESSION_METRICS = {
        "Mean Absolute Error": MeanAbsoluteError,
    }
    return REGRESSION_METRICS


def get_classifcation_metrics() -> Dict:
    CLASSIFICATION_METRICS = {

    }
    return CLASSIFICATION_METRICS
