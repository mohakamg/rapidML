REGRESSOR = "regressor"
CLASSIFIER = "classifier"
TYPES = [REGRESSOR, CLASSIFIER]


class PredictorType:
    """
    This class defines the type of Predictor Possible.
    """

    def __init__(self, predictor_type: str):
        assert predictor_type in TYPES, "Invalid Predictor Type"
        self.type = predictor_type

    def __str__(self) -> str:
        return self.type


class Regressor(PredictorType):
    """
    This class defines a Regressor Type
    """

    def __init__(self):
        super().__init__(REGRESSOR)


class Classifier(PredictorType):
    """
    This class defines a Classifer Type
    """

    def __init__(self):
        super().__init__(CLASSIFIER)
