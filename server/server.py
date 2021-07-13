from flask import Flask, request
from framework.stock.regressors import get_supported_regressors

# create the flask routing object
app = Flask(__name__)


@app.route('/<name>', methods=['GET'])
def index(name):
    """
    To url's with the following extension, '/regressors' or '/classifiers', they are only allowed to
    accept GET requests.

    Case 1: /regressors

    returns all supported regressors

    Case 2: /classifiers

    returns all supported classifiers

    Case 3: /

    returns a dict of all supported classifiers and regressors

    :return:
    """
    supported_regressors = list(get_supported_regressors().keys())
    supported_classifiers = []

    if name == "":
        # Return both supported types
        return {'regressors': supported_regressors, 'classifiers': supported_classifiers}
    elif name == 'regressors':
        return supported_regressors
    elif name == 'classifiers':
        return supported_classifiers

    return f"{name} is not valid argument"


if __name__ == '__main__':
    """
    This function routes to the specified IP and port number:
    
    IP: 0.0.0.0
    port: 8080
    
    """
    app.run(debug=True, host='localhost', port=8080)
