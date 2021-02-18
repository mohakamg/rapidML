from typing import Dict, List

import onnxmltools
import pandas as pd
import numpy as np
from onnxconverter_common import FloatTensorType
from sklearn import linear_model

from framework.interfaces.metric import MetricsCalculator
from framework.interfaces.predictor import Predictor
from framework.stock.predictortype import Regressor


class LinearRegression(Predictor):
    """
    This class implements a linear regression model from the SKlearn library.
    """

    def __init__(self, data_x: pd.DataFrame, data_y: pd.DataFrame,
                 metrics: List[MetricsCalculator],  data_split: Dict = {},
                 model_params: Dict = {}, metadata: Dict = {}):
        """
        The constructor intializes the base params.
        """
        super().__init__(Regressor(), "sklearn", False,
                         data_x, data_y,
                         metrics, data_split,
                         model_params, metadata)

    def get_name(self) -> str:
        """
        This function returns the name of the Predictor.
        :return name: Name of the Predictor
        """
        name = f"{self.library} Linear Regression"
        return name

    def build_model(self, filtered_model_params: Dict) -> linear_model.LinearRegression:
        """
        This function initializes the Linear Regression model with the
        model params.
        :return model: The built model
        """
        model = linear_model.LinearRegression(
            fit_intercept=filtered_model_params["fit_intercept"],
            normalize=filtered_model_params["normalize"],
            positive=filtered_model_params["normalize"]
        )
        return model

    def train_model(self, model: linear_model.LinearRegression,
                    data_x_train: pd.DataFrame, data_y_train: pd.DataFrame,
                    data_x_val: pd.DataFrame, data_y_val: pd.DataFrame,
                    data_x_test: pd.DataFrame, data_y_test: pd.DataFrame) -> [linear_model.LinearRegression, Dict]:
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

        :return trained_model: The linear regression model trained
        :return metrics: Dictionary containing the metrics.
        """
        metrics = {}
        # print(f"Model Params before Training coef: {model.coef_} | intercept: {model.intercept_}")

        # Train the model
        trained_model = model.fit(data_x_train.values, data_y_train.values)
        print(f"Model Params After Training coef: {trained_model.coef_} | intercept: {trained_model.intercept_}")

        # Gather Train Predictions
        train_preds = trained_model.predict(data_x_train.values)
        for metric in self.metrics:
            metrics[f"train_{metric}"] = metric.compute(data_y_train.values, train_preds)

        # Get the predictions on the Val data
        validation_preds = model.predict(data_x_val.values)
        for metric in self.metrics:
            metrics[f"val_{metric}"] = metric.compute(data_y_val.values, validation_preds)

        # Train along with Validation Data
        combined_x_data = pd.concat([data_x_train, data_x_val],
                                    axis=0)
        combined_y_data = pd.concat([data_y_train, data_y_val],
                                    axis=0)
        trained_model = trained_model.fit(combined_x_data.values, combined_y_data.values)
        print(f"Model Params After Validation coef: {trained_model.coef_} | intercept: {trained_model.intercept_}")

        # Get the predictions on the test data
        test_preds = trained_model.predict(data_x_train.values)
        for metric in self.metrics:
            metrics[f"test_{metric}"] = metric.compute(np.array(data_y_train), test_preds)

        return trained_model, metrics

    def serialize_model(self, model: linear_model.LinearRegression):
        """
        This function serializes the model into an Onnx Format.
        :return serialized_model: Model serialized into an Onnx format.
        """
        onnx_model = onnxmltools.convert_sklearn(model=model,
                                     name=self.metadata.get("name"),
                                     initial_types=[('float_input', FloatTensorType([None, len(self.data_x.columns.values)]))])
        serialized_model = onnx_model.SerializeToString()
        return serialized_model

    @staticmethod
    def get_default_params() -> Dict:
        """
        This function returns the default parameters along with a description.
        :return default_params: Default values with Description.
        """
        default_params = {
            "fit_intercept": {
                "default_value": True,
                "description": "Whether to calculate the intercept for this model. "
                               "If set to False, no intercept will be used in calculations "
                               "(i.e. data is expected to be centered)."
            },
            "normalize": {
                "default_value": False,
                "description": "This parameter is ignored when ``fit_intercept`` is set to False. "
                               "If True, the regressors X will be normalized before regression by "
                               "subtracting the mean and dividing by the l2-norm."
            },
            "positive": {
                "default_value": False,
                "description": "When set to ``True``, forces the coefficients to be positive. "
                               "This option is only supported for dense arrays."
            }
        }

        return default_params
