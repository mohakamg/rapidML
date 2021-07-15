from typing import Dict, Tuple

import onnxmltools
import pandas as pd
import numpy as np
from onnxconverter_common import FloatTensorType
from sklearn.ensemble import RandomForestRegressor

from framework.interfaces.predictor import Predictor
from framework.stock.predictortype import Regressor


class RandomForest(Predictor):
    """
    This class implements a Random Forest regression model from the SKlearn library.
    """

    def __init__(self, data_x: pd.DataFrame, data_y: pd.DataFrame,
                 data_split: Dict = {},
                 model_params: Dict = {}, metadata: Dict = {}):
        """
        The constructor intializes the base params.
        """
        super().__init__(Regressor(), "sklearn",
                         data_x, data_y, data_split,
                         model_params, metadata)

    def get_name(self) -> str:
        """
        This function returns the name of the Predictor.
        :return name: Name of the Predictor
        """
        name = f"{self.library} Random Forest Regressor"
        return name

    @staticmethod
    def does_support_multiobjective() -> bool:
        """
        This function returns if the predictor supports multiple outputs
        or not.
        :return multioutput: Bool
        """
        multioutput = True
        return multioutput

    def build_model(self, filtered_model_params: Dict) -> RandomForestRegressor:
        """
        This function initializes the Linear Regression model with the
        model params.
        :return model: The built model
        """
        model = RandomForestRegressor(
            n_estimators=filtered_model_params["n_estimators"],
            max_depth=filtered_model_params["max_depth"],
            min_samples_split=filtered_model_params["min_samples_split"],
            min_samples_leaf=filtered_model_params["min_samples_leaf"],
            max_samples=filtered_model_params["max_samples"],
            min_weight_fraction_leaf=filtered_model_params["min_weight_fraction_leaf"],
            max_features=filtered_model_params["max_features"],
            max_leaf_nodes=filtered_model_params["max_leaf_nodes"],
            min_impurity_decrease=filtered_model_params["min_impurity_decrease"],
            min_impurity_split=filtered_model_params["min_impurity_split"],
            bootstrap=filtered_model_params["bootstrap"],
            oob_score=filtered_model_params["oob_score"],
            n_jobs=filtered_model_params["n_jobs"],
            random_state=filtered_model_params["random_state"],
            warm_start=filtered_model_params["warm_start"],
            ccp_alpha=filtered_model_params["ccp_alpha"]
        )
        return model

    def train_model(self, model: RandomForestRegressor,
                    data_x_train: pd.DataFrame, data_y_train: pd.DataFrame,
                    data_x_val: pd.DataFrame, data_y_val: pd.DataFrame,
                    data_x_test: pd.DataFrame, data_y_test: pd.DataFrame) -> Tuple[RandomForestRegressor, Dict]:
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
        model.fit(data_x_train, data_y_train)

        # Gather Train Predictions
        train_preds = model.predict(data_x_train)
        for metric in self.metrics:
            metrics[f"train_{metric}"] = metric.compute(np.array(data_y_train), train_preds)

        # Get the predictions on the Val data
        validation_preds = model.predict(data_x_val)
        for metric in self.metrics:
            metrics[f"val_{metric}"] = metric.compute(np.array(data_y_val), validation_preds)

        # Train along with Validation Data
        combined_x_data = pd.concat([data_x_train, data_x_val],
                                    axis=0)
        combined_y_data = pd.concat([data_y_train, data_y_val],
                                    axis=0)
        model.fit(combined_x_data, combined_y_data)

        # Get the predictions on the test data
        test_preds = model.predict(data_x_train)
        for metric in self.metrics:
            metrics[f"test_{metric}"] = metric.compute(np.array(data_y_train), test_preds)

        return model, metrics

    def serialize_model(self, model: RandomForestRegressor):
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
            "n_estimators": {
                "default_value": 100,
                "description": "The number of trees in the forest.",
                "type": "int"
            },
            "criterion": {
                "default_value": "mse",
                "description": "The function to measure the quality of a split. "
                               "Supported criteria are “mse” for the mean squared error, "
                               "which is equal to variance reduction as feature selection "
                               "criterion, and “mae” for the mean absolute error. Options: {'mse', 'mae'}",
                "type": ['mse', 'mae']
            },
            "max_depth": {
                "default_value": 100,
                "description": "The maximum depth of the tree. "
                               "If None, then nodes are expanded until all "
                               "leaves are pure or until all leaves contain less than "
                               "min_samples_split samples.",
                "type": "int"
            },
            "min_samples_split": {
                "default_value": 2.0,
                "description": "The minimum number of samples required to split an internal node.",
                "type": "float"
            },
            "min_samples_leaf": {
                "default_value": 1.0,
                "description": '''
                The minimum number of samples required to be at a leaf node. 
                A split point at any depth will only be considered if it 
                leaves at least min_samples_leaf training samples in each of the left and right branches. 
                This may have the effect of smoothing the model, especially in regression.

                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) 
                are the minimum number of samples for each node.
                ''',
                "type": "float"
            },
            "min_weight_fraction_leaf": {
                "default_value": 0.0,
                "description": "The minimum weighted fraction of the sum total of"
                               " weights (of all the input samples) required to be at a leaf node. "
                               "Samples have equal weight when sample_weight is not provided.",
                "type": "float"
            },
            "max_features": {
                "default_value": "auto",
                "description": '''
                The number of features to consider when looking for the best split:

                If int, then consider max_features features at each split.
                If float, then max_features is a fraction and round(max_features * n_features) features are considered
                at each split.
                If “auto”, then max_features=n_features.
                If “sqrt”, then max_features=sqrt(n_features).
                If “log2”, then max_features=log2(n_features).
                If None, then max_features=n_features.
                Note: the search for a split does not stop until at least one valid partition of the node samples 
                is found, even if it requires to effectively inspect more than max_features features.
                ''',
                "type": ["auto", "sqrt", "log2"]
            },
            "max_leaf_nodes": {
                "default_value": 100,
                "description": "Grow trees with max_leaf_nodes in best-first fashion. "
                               "Best nodes are defined as relative reduction in impurity. "
                               "If None then unlimited number of leaf nodes.",
                "type": "int"
            },
            "min_impurity_decrease": {
                "default_value": 0.0,
                "description": '''
                A node will be split if this split induces a decrease of the impurity greater than or equal 
                to this value.

                The weighted impurity decrease equation is the following:
                
                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)
                where N is the total number of samples, N_t is the number of samples at the current node, 
                N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.
                
                N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.
                ''',
                "type": "float"
            },
            "bootstrap": {
                "default_value": True,
                "description": "Whether bootstrap samples are used when building trees. "
                               "If False, the whole dataset is used to build each tree.",
                "type": "bool"
            },
            "oob_score": {
                "default_value": False,
                "description": "whether to use out-of-bag samples to estimate the R^2 on unseen data",
                "type": "bool"
            },
            "n_jobs": {
                "default_value": 1,
                "description": "The number of jobs to run in parallel. fit, predict, "
                               "decision_path and apply are all parallelized over the trees. "
                               "None means 1 unless in a joblib.parallel_backend context. "
                               "-1 means using all processors. See Glossary for more details.",
                "type": "int"
            },
            "random_state": {
                "default_value": 0,
                "description": "Controls both the randomness of the bootstrapping of the samples "
                               "used when building trees (if bootstrap=True) and the sampling of the "
                               "features to consider when looking for the best split at each "
                               "node (if max_features < n_features)",
                "type": "int"
            },
            "warm_start": {
                "default_value": False,
                "description": "When set to True, reuse the solution of the previous call to "
                               "fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.",
                "type": "bool"
            },
            "ccp_alpha": {
                "default_value": 0.0,
                "description": "Complexity parameter used for Minimal Cost-Complexity Pruning. "
                               "The subtree with the largest cost complexity that is smaller than "
                               "ccp_alpha will be chosen. By default, no pruning is performed. ",
                "type": "float"
            },
            "max_samples": {
                "default_value": 1.0,
                "description": '''
                If bootstrap is True, the number of samples to draw from X to train each base estimator.
                
                If None (default), then draw X.shape[0] samples.
                If int, then draw max_samples samples.
                If float, then draw max_samples * X.shape[0] samples. 
                Thus, max_samples should be in the interval (0, 1).
                ''',
                "type": "float"
            }
        }

        return default_params
