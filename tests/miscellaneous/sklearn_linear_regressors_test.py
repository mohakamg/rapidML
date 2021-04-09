import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as rt

# Generate Dummy Data
from mpl_toolkits.mplot3d import Axes3D

from framework.metrics.mean_absolute_error import MeanAbsoluteError
from framework.predictors.train.regressors.executor import Executor
from framework.predictors.train.regressors.sklearn.linear_regression import LinearRegression
from framework.predictors.train.regressors.sklearn.random_forest import RandomForest
from framework.predictors.train.regressors.xgboost import XGBoost

num_points = 50
input_space_dimensionality = 2
output_space_dimensionality = 2

data_x = []
for _ in range(input_space_dimensionality):
    data_x.append(np.linspace(0, 5, num_points))
data_x_df = pd.DataFrame(data_x).T


# Generate Input space features
noise = 0.5

data_y = {f"feature_{val}": 2 * np.sin(data_x_df[0]) ** 2 + np.sin(data_x_df[1]) ** 2 + np.random.normal(5, noise, num_points) for val in range(output_space_dimensionality)}


# Convert features Dictionary into dataframe
data_y_df = pd.DataFrame(data_y)

# Train the Sklearn Predictor
metrics = [MeanAbsoluteError()]
executor = Executor(
    XGBoost, data_x_df, data_y_df, {}, {}, metrics, "Linear Regressor on Noisy Data"
)
executor.execute()

# Get the serialized model
predictors = executor.get_predictor_objects()
serialized_model = predictors[0].get_trained_model_bytes()

# Save the Model Locally
with open("test_model.onnx", "wb") as f:
    f.write(serialized_model)

# Load the Saved Model
sess = rt.InferenceSession("test_model.onnx")

# Plot the predictions
if input_space_dimensionality + output_space_dimensionality == 2:
    # Assemble the inputs
    features_dict = {'float_input': data_x_df.values.astype("float32")}
    pred_onx = sess.run([var.name for var in sess.get_outputs()], features_dict)[0]
    print("Error After Loading: ", metrics[0].compute(data_y_df.values, pred_onx))

    plt.scatter(data_x_df.values, data_y_df.values)
    plt.plot(data_x_df.values, pred_onx)
    plt.show()
elif input_space_dimensionality + output_space_dimensionality == 3:

    if input_space_dimensionality == 2:
        X, Y = np.meshgrid(*data_x)
        all_pts_df = pd.DataFrame(np.array([X, Y]).reshape(2, -1).T)

        # Assemble the inputs
        features_dict = {'float_input': all_pts_df.values.astype("float32")}
        pred_onx = sess.run([var.name for var in sess.get_outputs()], features_dict)[0]

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(data_x_df[0].values.reshape(1, -1).tolist()[0], data_x_df[1].values.reshape(1, -1).tolist()[0], data_y_df.values.reshape(1, -1).tolist()[0])
        ax.plot_trisurf(all_pts_df[0].values.reshape(1, -1).tolist()[0], all_pts_df[1].values.reshape(1, -1).tolist()[0], pred_onx.reshape(1, -1).tolist()[0], cmap='binary')
        fig.show()