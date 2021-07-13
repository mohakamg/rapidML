import json

import requests


class ServerTest:
    def __init__(self, url='http://localhost:8080/'):
        self.url = url

    def get_supported_regressors_test(self):
        # Construct the URL and message to send to the microservice
        endpoint = self.url+'regressors'

        print("\n          /get_supported_regressors Test           \n")

        # Using the requests library, send a POST request with the following
        # message to the URL.
        response = requests.get(url=endpoint)

        # Extract the status code and returned text
        status_code = response.status_code
        response_body = response.text

        if status_code == 200:
            print("Test Passed!")
            print(f"response: {response_body}\n")
        else:
            print(f"Test Failed with {status_code} status code.")
            print(f"response: {response_body}\n")

        return json.loads(response_body)


if __name__ == "__main__":
    test_obj = ServerTest()
    response = test_obj.get_supported_regressors_test()

    print(f"Supported Regressors: {response['body']['regressor']}")
