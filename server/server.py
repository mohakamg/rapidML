from flask import Flask, request
from framework.stock.regressors import get_supported_regressors

# create the flask routing object
app = Flask(__name__)


@app.route('/predictors/<name>', methods=['GET'])
def index(name):
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
    supported_classifiers = []

    if name == 'regressor':
        return {"regressor": supported_regressors}
    elif name == 'classifier':
        return {"classifier": supported_classifiers}

    return f"{name} is not valid argument"


if __name__ == '__main__':
    """
    This function routes to the specified IP and port number:
    
    IP: 0.0.0.0
    port: 8080
    
    """
    app.run(debug=True, host='localhost', port=8080)
