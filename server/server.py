from flask import Flask, request
from flask_cors import CORS
import pandas as pd
import uuid
from framework.predictors.train.regressors.executor import Executor
from framework.stock.classifiers import get_supported_classifiers
from framework.stock.regressors import get_supported_regressors
from framework.stock.metrics import get_regression_metrics, get_classifcation_metrics

# create the flask routing object
app = Flask(__name__)
CORS(app)

@app.route('/predictors/<name>', methods=['GET'])
def predictors(name):
    """
    This route returns supported predictors that fall under specific categories.

    Case 1: predictors/regressor

    RESPONSE: 
    {
        regressor: []
    }

    Case 2: predictors/classifier

    RESPONSE: 
    {
        classifier: []
    }


    :return: A dictionary containing either the supported regressors or classifiers.
    """
    supported_regressors = list(get_supported_regressors().keys())
    supported_classifiers = list(get_supported_classifiers().keys())

    if name == 'regressor':
        return {"regressor": supported_regressors}
    elif name == 'classifier':
        return {"classifier": supported_classifiers}

    return f"{name} is not valid argument"


@app.route('/metrics/<name>', methods=['GET'])
def metrics(name):
    """
    This route returns supported metrics for specific types of predictors.

    Case 1: metrtics/regressor

    RESPONSE: 
    {
        regressor: []
    }

    Case 2: metrics/classifier

    RESPONSE: 
    {
        classifier: []
    }

    :return: A dictionary containing either the supported regressors or classifiers.
    """
    supported_regressors = list(get_regression_metrics().keys())
    supported_classifiers = list(get_classifcation_metrics().keys())

    if name == 'regressor':
        return {"regressor": supported_regressors}
    elif name == 'classifier':
        return {"classifier": supported_classifiers}

    return f"{name} is not valid argument"


@app.route('/default_params/<pred_type>/<pred_name>', methods=['GET'])
def default_params(pred_type, pred_name):
    """
    
    This function returns a json with all default params for specific
    predictors.

    Example:
    route call: /default_params/regressor/Linear%20Regressor

    RESPONSE:
    {
        "fit_intercept": {
            "default_value": True,
            "description": "Whether to calculate the intercept for this model. "
                            "If set to False, no intercept will be used in calculations "
                            "(i.e. data is expected to be centered).",
            "type": "bool"
        },
        "normalize": {
            "default_value": False,
            "description": "This parameter is ignored when ``fit_intercept`` is set to False. "
                            "If True, the regressors X will be normalized before regression by "
                            "subtracting the mean and dividing by the l2-norm.",
            "type": "bool"
        },
        "positive": {
            "default_value": False,
            "description": "When set to ``True``, forces the coefficients to be positive. "
                            "This option is only supported for dense arrays.",
            "type": "bool"
        }
    }

    """
    predictor = None

    if pred_type == "regressor":
        predictor = get_supported_regressors().get(pred_name, None)
    elif pred_type == 'classifier':
        predictor = get_supported_classifiers().get(pred_name, None)

    if predictor is not None:
        return predictor.get_default_params()
    else:
        return f'{pred_type} and {pred_name} do not point to an implemented predictor.'


@app.route('/train', methods=['POST'])
def train_route():
    """
    This function kicks off training locally.

    REQUEST:
    {
        "data_path": "dataset.csv",
        "output_path": "./output_data",
        "config": {
            "predictor_type": "regressor",
            "predictor_name": "XGBoost",
            "cao_mapping": {
                "context": [],
                "actions": [],
                "outcomes": []
            },
            "data_split": {},
            "model_params": {},
            "metrics": "Mean Absolute Error",
            "model_metadata": {}
        }
    }

    RESPONSE:

    If Successful:
    "Training Complete"
    
    If Unsuccessful:
    "No Valid Predictor Selected."

    """
    request_json = request.json
    config = request_json['config']
    data_path = request_json['data_path']
    output_path = request_json['output_path']

    dataset = pd.read_csv(data_path)

    predictor_type = config['predictor_type']
    predictor_name = config['predictor_name']

    predictor = None

    if predictor_type == "regressor":
        predictor = get_supported_regressors().get(predictor_name, None)
    elif predictor_type == 'classifier':
        predictor = get_supported_classifiers().get(predictor_name, None)

    if predictor is not None:

        executor = Executor(predictor_class_ref=predictor,
                            data=dataset,
                            coa_mapping=config['cao_mapping'],
                            data_split=config['data_split'],
                            model_params=config['model_params'],
                            metrics=config['metrics'],
                            executor_name=str(uuid.uuid4()),
                            model_metadata=config['model_metadata']
                            )

        executor.execute()

        return "Training Complete"
    else:
        return 'No Valid Predictor Selected.'


if __name__ == '__main__':
    """
    This function routes to the specified IP and port number:
    
    IP: localhost or 127.0.0.1
    port: 8080
    
    """
    app.run(host='localhost', port=8080)
