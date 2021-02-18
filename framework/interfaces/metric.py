from typing import Dict

from framework.stock.predictortype import PredictorType


class MetricsCalculator:
    """
    This class implements a contract for implementations
    to calculate a certain kind of metric.
    """

    def __init__(self, predictor_type: PredictorType,
                 library: str, metric_params: Dict = None):
        """
        The constructor initializes the parameters.
        """
        self.predictor_type = predictor_type
        self.library = library
        self.metric_params = metric_params

    def get_name(self) -> str:
        """
        This function returns the name of the metric.
        :return name: String
        """
        raise NotImplementedError

    def __str__(self):
        return self.get_name()

    def compute(self, ground_truth, predictions, filtered_metric_params: Dict = {}):
        """
        This function computes the metrics.
        :param ground_truth: Array or DataFrame
        :param predictions: Array or DataFrame of the same format as ground_truth
        :param filtered_metric_params: Dictionary containing the parameters of the metric if any
        :return measurement: A score representing the metric.
        """
        raise NotImplementedError

    @staticmethod
    def get_default_params() -> Dict:
        """
        This function should return the default Parameters of the metric as Dictionary along with a description.
        :return default_model_params: Dictionary
        Format: {
            "parameter_name": {
                "default_value": "",
                "description": ""
            },
        }
        """
        raise NotImplementedError