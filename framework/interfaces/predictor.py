from typing import Dict, List, Tuple
import pandas as pd

from framework.interfaces.metric import MetricsCalculator
from framework.stock.predictortype import PredictorType


class Predictor:
    """
    This class contains the contract that any predictor
    must implement.
    """

    def __init__(self, predictor_type: PredictorType,
                 library: str,
                 data: pd.DataFrame, coa_mapping: Dict = {},
                 data_split: Dict = {},
                 model_params: Dict = {},
                 metadata: Dict = {}):
        """
        The constructor initializes the predictor, its params and the metadata.
        :param predictor_type: A predictor type that inherits from PredictorType
        :param library: Library this predictor belongs to
        :param data: DataFrame containing all processed data
        :param coa_mapping: Dictionary containing the mapping of indices from data to context, action, or
        outcome.
        :param data_split: Dictionary containing the training splits indexed by "train_pct" and "val_pct".
        :param model_params: Parameters of the model
        :param metadata: Dictionary describing any other information that must be stored along with the
        model. This might help in uniquely identifying the model
        :returns nothing
        """
        self.type = predictor_type
        self.library = library

        # Call data split function here
        self.data_x, self.data_y = self.get_data_xy_split(data, coa_mapping)

        self.data_split = data_split

        self.data_X_train, self.data_Y_train, \
        self.data_X_val, self.data_Y_val, \
        self.data_X_test, self.data_Y_test = self._generate_data_split()

        self.metrics: List[MetricsCalculator] = []

        self.model_params = model_params
        self.metadata = metadata

        # Internal Parameters that are used to store the
        # latest state of the model.
        self._trained_model = None
        self._serialized_bytes: bytes = b""

    def save_trained_model_state(self, trained_model) -> None:
        """
        This function stores the state of the trained model.
        :param trained_model: Trained model
        :return Nothing:
        """
        self._trained_model = trained_model

    def save_trained_model_bytes(self, model_bytes) -> None:
        """
        This function stores the state of the trained model.
        :param model_bytes: Trained model bytes
        :return Nothing:
        """
        self._serialized_bytes = model_bytes

    def get_trained_model(self):
        """
        This function returns the trained model if it exists.
        :return self._trained_model:
        """
        if self._trained_model:
            return self._trained_model

    def get_trained_model_bytes(self) -> bytes:
        """
        This function returns the serialized model if it exists.
        :return self._serialized_bytes:
        """
        if self._serialized_bytes:
            return self._serialized_bytes

    def register_metric(self, metric: MetricsCalculator) -> None:
        """
        This class registers the metric by type.
        :param metric: An Object of MetricsCalculator
        :returns: Nothing
        """
        self.metrics.append(metric)

    def register_metric_by_name(self, metric: str) -> bool:
        """
        This function Adds a metric.
        :param metric: String Describing the Metric.
        :return registered: Notifies if the metric was registered successfully
        """
        registered = False
        supported_metrics = self.type.fetch_supported_metrics()
        if metric.lower() in supported_metrics:
            self.register_metric(supported_metrics[metric.lower()])
            registered = True
        return registered

    def get_library(self) -> str:
        """
        This function returns the library that the predictor belongs to
        :return library: self.library
        """
        library = self.library
        return library

    @staticmethod
    def does_support_multiobjective() -> bool:
        """
        This function returns if the predictor supports multiple outputs
        or not.
        :return multioutput: Bool
        """
        raise NotImplementedError

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

    def export_model(self, op_path):
        """
        
        :param op_path: 
        :return: 
        """
        # Save the Model Locally
        with open(op_path, "wb") as f:
            f.write(self._serialized_bytes)

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

    @staticmethod
    def get_data_xy_split(data: pd.DataFrame,
                          coa_mapping: Dict = {}) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        This function takes a dataframe and a dictionary mapping indices to context,
        action, or outcome. This then splits the dataframe into two dataframes based
        on it's COA tagging.

        data_x: Context and Actions
        data_y: Outcomes

        :param data:
        :param coa_mapping:
        :return: A tuple containing two dataframes: data_x and data_y
        """
        
        data_x = data[coa_mapping["context"] + coa_mapping["actions"]]
        data_y = data[coa_mapping["outcomes"]]

        return data_x, data_y

    def __str__(self):
        return self.get_name()



