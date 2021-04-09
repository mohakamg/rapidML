import unittest
import os
import numpy as np
import pandas as pd
import onnxruntime as rt

from framework.stock.regressors import get_supported_regressors
from framework.metrics.mean_absolute_error import MeanAbsoluteError
from framework.predictors.train.regressors.executor import Executor


class RegressorRegressionTest(unittest.TestCase):
    """
    This class runs regression testing for all the
    regressors listing in Stock -> Regressors
    """

    def setUp(self):
        """
        This function fetches the availaible predictors.
        """
        self.availaible_predictors = get_supported_regressors()

    def generate_dummy_data(self, num_points,
                            input_space_dimensionality,
                            output_space_dimensionality, mode="linear"):
        """
        This function generates dummy data that can be used for testing.
        :param num_points: Number of points to generate
        :param input_space_dimensionality: Number of input features
        :param output_space_dimensionality: Number of Output Features
        :param mode: "linear" or "sine"
        :return data_x_df, data_y_df: Dataframes containing the data.
        """
        data_x = []
        for _ in range(input_space_dimensionality):
            data_x.append(np.linspace(0, 5, num_points))
        data_x_df = pd.DataFrame(data_x).T
        # Generate Input space features
        noise = 0.5

        if mode == "linear":
            col_sum = None
            for col in data_x_df.columns.values:
                if col_sum is None:
                    col_sum = data_x_df[col]
                else:
                    col_sum += data_x_df[col]

            data_y = {
                f"feature_{val}": 2 * col_sum + np.random.normal(5, noise, num_points) \
                for val in range(output_space_dimensionality)
            }
        elif mode == "sine":
            col_sum = None
            for col in data_x_df.columns.values:
                if col_sum is None:
                    col_sum = np.sin(data_x_df[col]) ** 2
                else:
                    col_sum += np.sin(data_x_df[col]) ** 2

            data_y = {
                f"feature_{val}": col_sum + np.random.normal(5, noise, num_points) \
                for val in range(output_space_dimensionality)
            }

        # Convert features Dictionary into dataFrame
        data_y_df = pd.DataFrame(data_y)

        return data_x_df, data_y_df

    def train(self, data_x_df, data_y_df):
        """
        This function is used to test the model
        :param data_x_df: Dataframe containing the input data
        :param data_y_df: Dataframe containing the output data
        :return nothing:
        """
        metrics = [MeanAbsoluteError()]
        for predictor_name, predictor in self.availaible_predictors.items():
            print(f"Evaluating Predictor: ", predictor_name)
            executor = Executor(
                predictor, data_x_df, data_y_df, {}, {}, metrics, "", {}
            )
            executor.execute()

    def test_training_1D_input_1D_output_linear(self):
        """
        This function trains the predictor considering only
        1 input and 1 output and knowing that
        """
        num_points = 50
        input_space_dimensionality = 1
        output_space_dimensionality = 1

        data_x_df, data_y_df = self.generate_dummy_data(num_points,
                                                        input_space_dimensionality,
                                                        output_space_dimensionality, "linear")
        self.train(data_x_df, data_y_df)

    def test_training_1D_input_1D_output_sine(self):


        num_points = 50
        input_space_dimensionality = 1
        output_space_dimensionality = 1

        data_x_df, data_y_df = self.generate_dummy_data(num_points,
                                                        input_space_dimensionality,
                                                        output_space_dimensionality, "sine")
        self.train(data_x_df, data_y_df)

    def test_training_2D_input_1D_output_linear(self):

        num_points = 50
        input_space_dimensionality = 2
        output_space_dimensionality = 1

        data_x_df, data_y_df = self.generate_dummy_data(num_points,
                                                        input_space_dimensionality,
                                                        output_space_dimensionality,
                                                        "linear")
        self.train(data_x_df, data_y_df)

    def test_training_2D_input_1D_output_sine(self):

        num_points = 50
        input_space_dimensionality = 2
        output_space_dimensionality = 1

        data_x_df, data_y_df = self.generate_dummy_data(num_points,
                                                        input_space_dimensionality,
                                                        output_space_dimensionality, "sine")
        self.train(data_x_df, data_y_df)

    def test_training_1D_input_2D_output_linear(self):

        num_points = 50
        input_space_dimensionality = 1
        output_space_dimensionality = 2

        data_x_df, data_y_df = self.generate_dummy_data(num_points,
                                                        input_space_dimensionality,
                                                        output_space_dimensionality,
                                                        "linear")
        self.train(data_x_df, data_y_df)

    def test_training_1D_input_2D_output_sine(self):

        num_points = 50
        input_space_dimensionality = 1
        output_space_dimensionality = 2

        data_x_df, data_y_df = self.generate_dummy_data(num_points,
                                                        input_space_dimensionality,
                                                        output_space_dimensionality, "sine")
        self.train(data_x_df, data_y_df)

    def test_training_2D_input_2D_output_linear(self):

        num_points = 50
        input_space_dimensionality = 2
        output_space_dimensionality = 2

        data_x_df, data_y_df = self.generate_dummy_data(num_points,
                                                        input_space_dimensionality,
                                                        output_space_dimensionality,
                                                        "linear")
        self.train(data_x_df, data_y_df)

    def test_training_2D_input_2D_output_sine(self):

        num_points = 50
        input_space_dimensionality = 2
        output_space_dimensionality = 2

        data_x_df, data_y_df = self.generate_dummy_data(num_points,
                                                        input_space_dimensionality,
                                                        output_space_dimensionality, "sine")
        self.train(data_x_df, data_y_df)

    def test_training_10D_input_10D_output_linear(self):

        num_points = 50
        input_space_dimensionality = 10
        output_space_dimensionality = 10

        data_x_df, data_y_df = self.generate_dummy_data(num_points,
                                                        input_space_dimensionality,
                                                        output_space_dimensionality,
                                                        "linear")
        self.train(data_x_df, data_y_df)

    def test_training_10D_input_10D_output_sine(self):

        num_points = 50
        input_space_dimensionality = 10
        output_space_dimensionality = 10

        data_x_df, data_y_df = self.generate_dummy_data(num_points,
                                                        input_space_dimensionality,
                                                        output_space_dimensionality, "sine")
        self.train(data_x_df, data_y_df)

    def test_onnx_inference_10D_input_10D_output_sine(self):

        num_points = 50
        input_space_dimensionality = 10
        output_space_dimensionality = 10

        data_x_df, data_y_df = self.generate_dummy_data(num_points,
                                                        input_space_dimensionality,
                                                        output_space_dimensionality, "sine")
        self.train(data_x_df, data_y_df)


        metrics = [MeanAbsoluteError()]
        for predictor_name, predictor in self.availaible_predictors.items():
            print(f"Evaluating Predictor: ", predictor_name)
            executor = Executor(
                predictor, data_x_df, data_y_df, {}, {}, metrics, "", {}
            )
            executor.execute()

            # Loop over the predictors
            for predictor in executor.get_predictor_objects():

                save_path = f"{predictor_name}.onnx"
                # Save the Model Locally
                with open(save_path, "wb") as f:
                    f.write(predictor.get_trained_model_bytes())

                # Load the Saved Model
                sess = rt.InferenceSession(save_path)

                # Assemble the inputs
                sample_data = {'float_input': data_x_df.values.astype("float32")}

                pred_onx = sess.run([var.name for var in sess.get_outputs()], sample_data)[0]
                print(pred_onx)

                os.remove(save_path)

if __name__ == '__main__':
    unittest.main()