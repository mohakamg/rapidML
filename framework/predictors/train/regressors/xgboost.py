from typing import Dict, List

import pandas as pd
import numpy as np
from onnxconverter_common import FloatTensorType
from xgboost import XGBRegressor
import onnxmltools

from framework.interfaces.metric import MetricsCalculator
from framework.interfaces.predictor import Predictor
from framework.stock.predictortype import Regressor
from utils.onnx_utils import convert_dataframe_schema


class XGBoost(Predictor):
    """
    This class implements a Random Forest regression model from the SKlearn library.
    """

    def __init__(self, data_x: pd.DataFrame, data_y: pd.DataFrame,
                 metrics: List[MetricsCalculator],  data_split: Dict = {},
                 model_params: Dict = {}, metadata: Dict = {}):
        """
        The constructor intializes the base params.
        """
        super().__init__(Regressor(), "xgboost", True,
                         data_x, data_y,
                         metrics, data_split,
                         model_params, metadata)

    def get_name(self) -> str:
        """
        This function returns the name of the Predictor.
        :return name: Name of the Predictor
        """
        name = f"{self.library} XGBoost Regressor"
        return name

    def build_model(self, filtered_model_params: Dict) -> XGBRegressor:
        """
        This function initializes the Linear Regression model with the
        model params.
        :return model: The built model
        """
        model = XGBRegressor(
            max_depth=filtered_model_params["max_depth"],
            learning_rate=filtered_model_params["learning_rate"],
            n_estimators=filtered_model_params["n_estimators"],
            verbosity=filtered_model_params["verbosity"],
            # objective=filtered_model_params["objective"],
            booster=filtered_model_params["booster"],
            tree_method=filtered_model_params["tree_method"],
            n_jobs=filtered_model_params["n_jobs"],
            gamma=filtered_model_params["gamma"],
            min_child_weight=filtered_model_params["min_child_weight"],
            max_delta_step=filtered_model_params["max_delta_step"],
            subsample=filtered_model_params["subsample"],
            colsample_bytree=filtered_model_params["colsample_bytree"],
            colsample_bylevel=filtered_model_params["colsample_bylevel"],
            colsample_bynode=filtered_model_params["colsample_bynode"],
            reg_alpha=filtered_model_params["reg_alpha"],
            reg_lambda=filtered_model_params["reg_lambda"],
            scale_pos_weight=filtered_model_params["scale_pos_weight"],
            base_score=filtered_model_params["base_score"],
            random_state=filtered_model_params["random_state"],
            missing=np.nan,
            num_parallel_tree=filtered_model_params["num_parallel_tree"],
            monotone_constraints=filtered_model_params["monotone_constraints"],
            interaction_constraints=filtered_model_params["interaction_constraints"],
            importance_type=filtered_model_params["importance_type"]
        )
        return model

    def train_model(self, model: XGBRegressor,
                    data_x_train: pd.DataFrame, data_y_train: pd.DataFrame,
                    data_x_val: pd.DataFrame, data_y_val: pd.DataFrame,
                    data_x_test: pd.DataFrame, data_y_test: pd.DataFrame) -> [XGBRegressor, Dict]:
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

        # Train the model
        model.fit(data_x_train.values, data_y_train.values)

        # Gather Train Predictions
        train_preds = model.predict(data_x_train.values)
        for metric in self.metrics:
            metrics[f"train_{metric}"] = metric.compute(np.array(data_y_train), train_preds)

        # Get the predictions on the Val data
        validation_preds = model.predict(data_x_val.values)
        for metric in self.metrics:
            metrics[f"val_{metric}"] = metric.compute(np.array(data_y_val), validation_preds)

        # Train along with Validation Data
        combined_x_data = pd.concat([data_x_train, data_x_val],
                                    axis=0)
        combined_y_data = pd.concat([data_y_train, data_y_val],
                                    axis=0)
        model.fit(combined_x_data.values, combined_y_data.values)

        # Get the predictions on the test data
        test_preds = model.predict(data_x_train.values)
        for metric in self.metrics:
            metrics[f"test_{metric}"] = metric.compute(np.array(data_y_train), test_preds)

        return model, metrics

    def serialize_model(self, model: XGBRegressor):
        """
        This function serializes the model into an Onnx Format.
        :return serialized_model: Model serialized into an Onnx format.
        """
        onnx_model = onnxmltools.convert_xgboost(model=model,
                                     name=self.metadata.get("name"),
                                     initial_types=[('float_input', FloatTensorType([None, len(self.data_x.columns.values)]))]
                                                 )
        serialized_model = onnx_model.SerializeToString()
        return serialized_model

    @staticmethod
    def get_default_params() -> Dict:
        """
        This function returns the default parameters along with a description.
        :return default_params: Default values with Description.
        """
        default_params = {
            "n_estimators": {
                "default_value": 100,
                "description": "Number of gradient boosted trees.  "
                               "Equivalent to number of boosting rounds."
            },
            "max_depth": {
                "default_value": None,
                "description": "Maximum tree depth for base learners."
            },
            "learning_rate": {
                "default_value": None,
                "description": "Boosting learning rate (xgb's 'eta')"
            },
            "verbosity": {
                "default_value": None,
                "description": "The degree of verbosity. Valid values are 0 (silent) - 3 (debug)."
            },
            "booster": {
                "default_value": None,
                "description": "Specify which booster to use: gbtree, gblinear or dart."
            },
            "tree_method": {
                "default_value": None,
                "description":
                    '''
                    Specify which tree method to use.  Default to auto.  If this parameter
                    is set to default, XGBoost will choose the most conservative option
                    available.  It's recommended to study this option from parameters
                    document.
                    '''
            },
            "n_jobs": {
                "default_value": None,
                "description": '''
                Number of parallel threads used to run xgboost.  When used with other Scikit-Learn
                algorithms like grid search, you may choose which algorithm to parallelize and
                balance the threads.  Creating thread contention will significantly slow dowm both
                algorithms.
                '''
            },
            "gamma": {
                "default_value": None,
                "description": "Minimum loss reduction required to make a further "
                               "partition on a leaf node of the tree."
            },
            "min_child_weight": {
                "default_value": None,
                "description": "Minimum loss reduction required to make a further "
                               "partition on a leaf node of the tree."
            },
            "max_delta_step": {
                "default_value": None,
                "description": "Maximum delta step we allow each tree's weight estimation to be."
            },
            "subsample": {
                "default_value": None,
                "description": "Subsample ratio of the training instance."
            },
            "colsample_bytree": {
                "default_value": None,
                "description": "Subsample ratio of columns when constructing each tree."
            },
            "colsample_bylevel": {
                "default_value": None,
                "description": "Subsample ratio of columns for each level."
            },
            "colsample_bynode": {
                "default_value": None,
                "description": "Subsample ratio of columns for each split."
            },
            "reg_alpha": {
                "default_value": None,
                "description": "L1 regularization term on weights"
            },
            "reg_lambda": {
                "default_value": None,
                "description": "L2 regularization term on weights"
            },
            "scale_pos_weight": {
                "default_value": None,
                "description": "Balancing of positive and negative weights."
            },
            "random_state": {
                "default_value": None,
                "description": "Random number seed."
            },
            "base_score": {
                "default_value": None,
                "description": "The initial prediction score of all instances, global bias."
            },
            "missing": {
                "default_value": None,
                "description": "Value in the data which needs to be present as a missing value."
            },
            "num_parallel_tree": {
                "default_value": None,
                "description": "Used for boosting random forest."
            },
            "monotone_constraints": {
                "default_value": None,
                "description": " Constraint of variable monotonicity.  "
                               "See tutorial for more information."
            },
            "interaction_constraints": {
                "default_value": None,
                "description": '''
                Constraints for interaction representing permitted interactions.  The
                constraints must be specified in the form of a nest list, e.g. [[0, 1],
                [2, 3, 4]], where each inner list is a group of indices of features
                that are allowed to interact with each other.  See tutorial for more
                information
                '''
            },
            "importance_type": {
                "default_value": "gain",
                "description": '''
                The feature importance type for the feature_importances. property:
                either "gain", "weight", "cover", "total_gain" or "total_cover".
                '''
            }
        }

        return default_params
