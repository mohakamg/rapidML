from typing import Dict

from framework.interfaces.metric import MetricsCalculator
from framework.stock.predictortype import Regressor


class MeanAbsoluteError(MetricsCalculator):
    """
    This class computes the Mean Absolute Error
    using the SKlearn library.
    """
    def __init__(self):
        """
        This function initializes the base class
        """
        super().__init__(Regressor(), "sklearn")

    def get_name(self) -> str:
        """
        This function returns the name of the metric.
        :return name: Name of the metric
        """
        name = "Mean Absolute Error"
        return name

    def compute(self, ground_truth, predictions, filtered_metric_params: Dict = {}):
        """
        This function computes the Mean absolute error.
        :param ground_truth: Array or DataFrame
        :param predictions: Array or DataFrame of the same format as ground_truth
        :param filtered_metric_params: Dictionary containing the parameters of the metric if any
        :return measurement: A score representing the metric.
        """
        from sklearn.metrics import mean_absolute_error
        measurement = mean_absolute_error(ground_truth, predictions,
                                          multioutput=filtered_metric_params.get("multioutput",
                                                                                 "uniform_average"))
        return measurement

    @staticmethod
    def get_default_params() -> Dict:
        """
        This function returns the default parameters for the sklearn MAE function.
        :return default_params: Dictionary containing description and default value for
        each param
        """
        default_params = {
            "multioutput": {
                "default_value": "uniform_average",
                "description":
                    '''
                        {'raw_values', 'uniform_average'}  or array-like of shape \
                        (n_outputs,), default='uniform_average'
                        Defines aggregating of multiple output values.
                        Array-like value defines weights used to average errors.
            
                        'raw_values': Returns a full set of errors in case of multioutput input.
                        'uniform_average': Errors of all outputs are averaged with uniform weight.
                    '''
            }
        }
        return default_params
