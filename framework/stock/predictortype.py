from typing import Dict, List

from framework.interfaces.metric import MetricsCalculator
from framework.stock.metrics import get_regression_metrics, get_classifcation_metrics

REGRESSOR = "regressor"
CLASSIFIER = "classifier"
TYPES = [REGRESSOR, CLASSIFIER]


class PredictorType:
    """
    This class defines the type of Predictor Possible.
    """

    def __init__(self, predictor_type: str):
        """
        The constructor confirms if the type of predictor is supported.
        :param predictor_type: String describing a name for the type of the
        predictor.
        """
        assert predictor_type in TYPES, "Invalid Predictor Type"
        self.type = predictor_type

    def __str__(self) -> str:
        """
        This function overrides the string representation of the
        class.
        :return self.type: String
        """
        return self.type

    def fetch_supported_metrics(self) -> Dict[str, MetricsCalculator]:
        """
        This function is responsible for returning the metrics
        available to this type of Predictor
        """
        raise NotImplementedError

class Regressor(PredictorType):
    """
    This class defines a Regressor Type
    """

    def __init__(self):
        """
        The constructor initializes the super class.
        """
        super().__init__(REGRESSOR)

    def fetch_supported_metrics(self) -> List[str]:
        """
        This function is responsible for returning the metrics
        available to this type of Predictor
        """
        regression_metrics = get_regression_metrics()
        return list(regression_metrics.keys())


class Classifier(PredictorType):
    """
    This class defines a Classifer Type
    """

    def __init__(self):
        """
        The constructor initializes the super class.
        """
        super().__init__(CLASSIFIER)

    def fetch_supported_metrics(self) -> List[str]:
        """
        This function is responsible for returning the metrics
        available to this type of Predictor
        """
        classification_metrics = get_classifcation_metrics()
        return list(classification_metrics.keys())
