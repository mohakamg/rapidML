import copy
from typing import List, Dict, Union
import pandas as pd

from framework.interfaces.metric import MetricsCalculator
from framework.interfaces.predictor import Predictor
from utils.overlayer import Overlayer
from utils.predictor import flatten_default_params


class Executor:
    """
    This class is responsible for executing the lifecycle of a Regression
    Predictor.
    """

    def __init__(self, predictor_class_ref,
                 data: pd.DataFrame, coa_mapping: Dict,
                 data_split: Dict,
                 model_params: Dict, metrics: Union[List[str], List[MetricsCalculator]],
                 executor_name: str, model_metadata: Dict = {}):
        """
        The constructor contains the state information of a predictor.
        """

        self.predictor_class_ref = predictor_class_ref
        self._predictors: List[Predictor] = self._create_predictors(
            data, coa_mapping, data_split, model_params, metrics, model_metadata
        )
        self.executor_name = executor_name

        # State variables
        self._executed = False

    def get_executor_name(self) -> str:
        """
        This function returns the name of the current executor.
        :return self.executor_name: Name
        """
        return self.executor_name

    def was_executed(self) -> bool:
        """
        This function determines if the class was executed.
        :return self_executed: Bool
        """
        return self._executed

    def get_predictor_objects(self) -> List[Predictor]:
        """
        This function returns the list of predictors that are part
        of this executor.
        :return self.predictors: List of Predictors
        """
        return self._predictors

    def _create_predictor(self, data: pd.DataFrame, coa_mapping: Dict,
                          data_split: Dict,
                          model_params: Dict, metrics: Union[List[str], List[MetricsCalculator]],
                          model_metadata: Dict = {}) -> Predictor:
        """
        This function creates the predictor
        :return predictor: Predictor with the desired output.
        """
        # Create the Predictor
        predictor: Predictor = self.predictor_class_ref(data, coa_mapping,
                            data_split,
                            model_params,
                            model_metadata)

        # Initialize the Metrics
        for metric in metrics:
            if isinstance(metric, str):
                predictor.register_metric_by_name(metric)
            else:
                predictor.register_metric(metric)

        return predictor

    def _create_predictors(self, data: pd.DataFrame, coa_mapping: Dict,
                           data_split: Dict,
                           model_params: Dict, metrics: Union[List[str], List[MetricsCalculator]],
                           model_metadata: Dict = {}) -> List[Predictor]:
        """
        This function is responsible for creating one or more predictors depending on
        the ability to support multiple outputs or not
        :param data:
        ::
        :param data_split:
        :param model_params:
        :param metrics:
        :param model_metadata:
        :return:
        """
        predictors = []
        if len(coa_mapping["outcomes"]) > 1 and self.predictor_class_ref.does_support_multiobjective():
            model_metadata_copy = copy.deepcopy(model_metadata)
            model_metadata_copy["inputs"] = coa_mapping["context"].tolist().extend(coa_mapping["actions"])
            model_metadata_copy["outputs"] = coa_mapping["outcomes"]
            model_metadata_copy["data_split"] = data_split
            predictor = self._create_predictor(
                data, coa_mapping,
                data_split,
                model_params,
                metrics,
                data_split
            )
            predictors = [predictor]
        else:
            model_metadata_copy = copy.deepcopy(model_metadata)
            model_metadata_copy["inputs"] = coa_mapping["context"] + coa_mapping["actions"]
            model_metadata_copy["data_split"] = data_split

            # Create one predictor per output
            for column in coa_mapping["outcomes"]:
                coa_mapping_per_outcome = copy.copy(coa_mapping)
                coa_mapping_per_outcome["outcomes"] = [column]
                model_metadata_copy["outputs"] = [column]
                predictor = self._create_predictor(
                    data,
                    coa_mapping_per_outcome,
                    data_split,
                    model_params,
                    metrics,
                    data_split
                )
                predictors.append(predictor)

        return predictors

    def execute(self):
        """
        This function executes the lifecycle of a predictor.
        :return serialized_models: Returns the list of serialized bytes of the model.
        :return metrics: Dictionary containing all the metrics evaluated.
        """
        # Overlay Users Config on Models default params
        default_config = self.predictor_class_ref.get_default_params()
        flattened_default_config = flatten_default_params(default_config)
        overlayed_config = Overlayer.overlay_configs(
            flattened_default_config, self._predictors[0].model_params
        )
        # Loop over the predictors
        for predictor in self._predictors:

            # Build the model
            model = predictor.build_model(overlayed_config)

            # Train the model
            trained_model, metrics = predictor.train_model(model,
                                                           predictor.data_X_train, predictor.data_Y_train,
                                                           predictor.data_X_val, predictor.data_Y_val,
                                                           predictor.data_X_test, predictor.data_Y_test)
            print(metrics)
            predictor.save_trained_model_state(trained_model)

            # Serialize Model
            serialized_bytes = predictor.serialize_model(trained_model)
            predictor.save_trained_model_bytes(serialized_bytes)









