from typing import Dict

import pandas as pd
import numpy as np
from onnxconverter_common import FloatTensorType
from xgboost import XGBRegressor
import onnxmltools

from framework.interfaces.predictor import Predictor
from framework.stock.predictortype import Regressor


class XGBoost(Predictor):
    """
    This class implements a Random Forest regression model from the SKlearn library.
    """

    def __init__(self, data: pd.DataFrame, coa_mapping: Dict = {},
                 data_split: Dict = {},
                 model_params: Dict = {}, metadata: Dict = {}):
        """
        The constructor intializes the base params.
        """
        super().__init__(Regressor(), "xgboost",
                         data, coa_mapping,
                         data_split,
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
            # missing=np.nan,
            num_parallel_tree=filtered_model_params["num_parallel_tree"],
            # monotone_constraints=filtered_model_params["monotone_constraints"],
            # interaction_constraints=filtered_model_params["interaction_constraints"],
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
    def does_support_multiobjective() -> bool:
        """
        This function returns if the predictor supports multiple outputs
        or not.
        :return multioutput: Bool
        """
        multioutput = False
        return multioutput

    @staticmethod
    def get_default_params() -> Dict:
        """
        This function returns the default parameters along with a description.

        Note:
        Params not included at the moment:
        - monotone_constraints
        - interaction_constraints
        - missing

        :return default_params: Default values with Description.
        """
        default_params = {
            "n_estimators": {
                "default_value": 100,
                "description": "Number of gradient boosted trees.  "
                               "Equivalent to number of boosting rounds.",
                "type": "int"
            },
            "max_depth": {
                "default_value": 6,
                "description": "Maximum tree depth for base learners.",
                "type": "int"
            },
            "learning_rate": {
                "default_value": 0.3,
                "description": "Boosting learning rate (xgb's 'eta')",
                "type": "float"
            },
            "verbosity": {
                "default_value": 1,
                "description": "The degree of verbosity. Valid values are 0 (silent) - 3 (debug).",
                "type": [0, 1, 2, 3]
            },
            "booster": {
                "default_value": "gbtree",
                "description": "Specify which booster to use: gbtree, gblinear or dart.",
                "type": ['gbtree', 'gblinear', 'dart']
            },
            "tree_method": {
                "default_value": "auto",
                "description":
                    '''
                    Specify which tree method to use.  Default to auto.  If this parameter
                    is set to default, XGBoost will choose the most conservative option
                    available.  It's recommended to study this option from parameters
                    document.
                    ''',
                "type": ["auto", "exact", "approx", "hist", "gpu_hist"]
            },
            "n_jobs": {
                "default_value": 1,
                "description": '''
                Number of parallel threads used to run xgboost.  When used with other Scikit-Learn
                algorithms like grid search, you may choose which algorithm to parallelize and
                balance the threads.  Creating thread contention will significantly slow dowm both
                algorithms.
                ''',
                "type": "int"
            },
            "gamma": {
                "default_value": 0.0,
                "description": "Minimum loss reduction required to make a further "
                               "partition on a leaf node of the tree.",
                "type": "float"
            },
            "min_child_weight": {
                "default_value": 1.0,
                "description": "Minimum loss reduction required to make a further "
                               "partition on a leaf node of the tree.",
                "type": "float"
            },
            "max_delta_step": {
                "default_value": 0.0,
                "description": "Maximum delta step we allow each tree's weight estimation to be.",
                "type": "float"
            },
            "subsample": {
                "default_value": 1.0,
                "description": "Subsample ratio of the training instance.",
                "type": "float"
            },
            "colsample_bytree": {
                "default_value": 1.0,
                "description": "Subsample ratio of columns when constructing each tree.",
                "type": "float"
            },
            "colsample_bylevel": {
                "default_value": 1.0,
                "description": "Subsample ratio of columns for each level.",
                "type": "float"
            },
            "colsample_bynode": {
                "default_value": 1.0,
                "description": "Subsample ratio of columns for each split.",
                "type": "float"
            },
            "reg_alpha": {
                "default_value": 0.0,
                "description": "L1 regularization term on weights",
                "type": "float"
            },
            "reg_lambda": {
                "default_value": 0.0,
                "description": "L2 regularization term on weights",
                "type": "float"
            },
            "scale_pos_weight": {
                "default_value": 1.0,
                "description": "Balancing of positive and negative weights.",
                "type": "float"
            },
            "random_state": {
                "default_value": 0,
                "description": "Random number seed.",
                "type": "int"
            },
            "base_score": {
                "default_value": 0.5,
                "description": "The initial prediction score of all instances, global bias.",
                "type": "float"
            },
            # "missing": {
            #     "default_value": None,
            #     "description": "Value in the data which needs to be present as a missing value.",
            #     "type": "float"
            # },
            "num_parallel_tree": {
                "default_value": 1,
                "description": "Used for boosting random forest.",
                "type": "int"
            },
            # "monotone_constraints": {
            #     "default_value": "(0,0)",
            #     "description": " Constraint of variable monotonicity.  "
            #                    "See tutorial for more information.",
            #     "type": "str"
            # },
            # "interaction_constraints": {
            #     "default_value": None,
            #     "description": '''
            #     Constraints for interaction representing permitted interactions.  The
            #     constraints must be specified in the form of a nest list, e.g. [[0, 1],
            #     [2, 3, 4]], where each inner list is a group of indices of features
            #     that are allowed to interact with each other.  See tutorial for more
            #     information
            #     ''',
            #     "type": "str"
            # },
            "importance_type": {
                "default_value": "gain",
                "description": '''
                The feature importance type for the feature_importances. property:
                either "gain", "weight", "cover", "total_gain" or "total_cover".
                ''',
                "type": ["gain", "weight", "cover", "total_gain", "total_cover"]
            }
        }

        return default_params
