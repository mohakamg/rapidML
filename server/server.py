from flask import Flask, request
import pandas as pd
import uuid
from framework.predictors.train.regressors.executor import Executor
from framework.stock.classifiers import get_supported_classifiers
from framework.stock.regressors import get_supported_regressors
from framework.stock.metrics import get_regression_metrics, get_classifcation_metrics

# create the flask routing object
app = Flask(__name__)


@app.route('/predictors/<name>', methods=['GET'])
def predictors(name):
    """
    To url's with the following extension, '/regressor' or '/classifier', they are only allowed to
    accept GET requests.

    Case 1: predictors/regressor

    returns all supported regressors

    Case 2: predictors/classifier

    returns all supported classifiers

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
    To url's with the following extension, '/regressor' or '/classifier', they are only allowed to
    accept GET requests.

    Case 1: metrtics/regressor

    returns all supported regressors

    Case 2: metrics/classifier

    returns all supported classifiers

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
    else:
        print('No Valid Predictor Selected.')
        exit(0)


if __name__ == '__main__':
    """
    This function routes to the specified IP and port number:
    
    IP: 0.0.0.0
    port: 8080
    
    """
    app.run(debug=True, host='localhost', port=8080)
