from typing import Dict, List
import pandas as pd

from framework.interfaces.metric import MetricsCalculator
from framework.stock.predictortype import PredictorType


class Predictor:
    """
    This class contains the contract that any predictor
    must implement.
    """

    def __init__(self, predictor_type: PredictorType,
                 library: str, multioutput: bool,
                 data_x: pd.DataFrame, data_y: pd.DataFrame,
                 metrics: List[MetricsCalculator],
                 data_split: Dict = {},
                 model_params: Dict = {},
                 metadata: Dict = {}):
        """
        The constructor initializes the predictor, its params and the metadata.
        :param predictor_type: A predictor type that inherits from PredictorType
        :param library: Library this predictor belongs to
        :param multioutput: Specifies if the predictor supports multioutput.
        :param data_x: DataFrame containing the processed input features
        :param data_y: DataFrame containing the processed output features
        :param data_split: Dictionary containing the training splits indexed by "train_pct" and "val_pct".
        :param model_params: Parameters of the model
        :param metadata: Dictionary describing any other information that must be stored along with the
        model.
        :returns nothing
        """
        self.type = predictor_type
        self.library = library
        self.multioutput = multioutput

        self.data_x = data_x
        self.data_y = data_y
        self.data_split = data_split
        self.metrics = metrics

        self.model_params = model_params
        self.metadata = metadata

    def get_library(self) -> str:
        """
        This function returns the library that the predictor belongs to
        :return library: self.library
        """
        library = self.library
        return library

    def does_support_multioutput(self) -> bool:
        """
        This function returns if the predictor supports multiple outputs
        or not.
        :return multioutput: Bool
        """
        multioutput = self.multioutput
        return multioutput

    def build_model(self, filtered_model_params: Dict):
        """
        This function must be overriden to build the model using the model
        parameters if desired and return a model.
        :param filtered_model_params: Dictionary containing the filtered model parameters
        after it has been overlayed with the defaults.
        :return model: The built model.
        """
        raise NotImplementedError

    def _validate_metrics(self):
        """
        This function validates if the metrics passed relate to the Predictor type
        :return: nothing
        """
        for metric in self.metrics:
            assert metric.predictor_type == self.type, f"Metric {metric} is not of type {self.type}"

    def _generate_data_split(self) -> List[pd.DataFrame]:
        """
        This function generates the data split.
        :return data_X_train, data_Y_train, data_X_val, data_Y_val, data_X_test, data_Y_test: Pandas DataFrame
        split into training, validation and test splits.
        """
        from sklearn.model_selection import train_test_split
        if not self.data_split:
            data_split = {"train_pct": 0.9, "val_pct": 0.05}
        else:
            data_split = self.data_split
        data_X_train, data_X_val, data_Y_train, data_Y_val = train_test_split(
            self.data_x, self.data_y,
            train_size=data_split["train_pct"]
        )
        data_X_val, data_X_test, data_Y_val, data_Y_test = train_test_split(
            data_X_val, data_Y_val,
            train_size=data_split["val_pct"]/(1-data_split["train_pct"])
        )
        return [
            data_X_train, data_Y_train,
            data_X_val, data_Y_val,
            data_X_test, data_Y_test
        ]

    def train_model(self, model,
                    data_x_train: pd.DataFrame, data_y_train: pd.DataFrame,
                    data_x_val: pd.DataFrame, data_y_val: pd.DataFrame,
                    data_x_test: pd.DataFrame, data_y_test: pd.DataFrame):
        """
        This function must be overriden to train the built model from the build_model step
        given the Data and must return the trained model and the desired metrics as a dictionary.
        :param model: The model built in the build_model step
        :param data_x_train: DataFrame containing the processed input features split for training
        :param data_y_train: DataFrame containing the processed output features split for training
        :param data_x_val: DataFrame containing the processed input features split for validation
        :param data_y_val: DataFrame containing the processed output features split for validation
        :param data_x_test: DataFrame containing the processed input features split for testing
        :param data_y_test: DataFrame containing the processed output features split for testing

        :return trained_model, metrics:
        """
        raise NotImplementedError

    def serialize_model(self, model):
        """
        This function serializes the model into an agreed format and returns it.
        :param model: A trained model from the previous train_model step.
        :return serialized_model: The serialized model bytes.
        """
        raise NotImplementedError

    @staticmethod
    def get_default_params() -> Dict:
        """
        This function should return the default Parameters of the model as Dictionary along with a description.
        :return default_model_params: Dictionary
        Format: {
            "parameter_name": {
                "default_value": "",
                "description": ""
            },
        }
        """
        raise NotImplementedError

    def get_name(self) -> str:
        """
        This function returns the name of the Predictor.
        :return name: String
        """
        raise NotImplementedError

    def __str__(self):
        return self.get_name()



