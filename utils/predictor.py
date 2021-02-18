from typing import Dict


def flatten_default_params(param_dict: Dict) -> Dict:
    """
    This function flattens the parameter dictionary implemented by the Predictor interface
    :param param_dict:
    Format: {
            "parameter_name": {
                "default_value": "",
                "description": ""
                },
            }
    :return flatten_dict: Flattened Dictionary
    Format : {
        "parameter_name": "default_value",
    }
    """
    flatten_dict = {param_name: value["default_value"] for param_name, value in param_dict.items()}
    return flatten_dict