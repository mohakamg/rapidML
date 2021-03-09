import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from framework.metrics.mean_absolute_error import MeanAbsoluteError
from framework.predictors.train.regressors.sklearn.random_forest import RandomForest
from utils.predictor import flatten_default_params

# Generate Dummy Data
num_points = 50
output_space_dimensionality = 1
data_y = np.linspace(0, 100, num_points)
data_y_df = pd.DataFrame(data_y)

# Generate Input space features
noise_std = 1.6
target_dict = {f"feature_{val}": 2 * data_y + np.random.normal(5, noise_std, num_points) for val in range(output_space_dimensionality)}

# Convert features Dictionary into dataframe
target_df = pd.DataFrame(target_dict)

# Train the Sklearn Predictor
metrics = [MeanAbsoluteError()]
sklearn_linear_regressor_predictor = RandomForest(target_df, data_y_df, metrics=metrics, metadata={"name": "test_linear_reg_SKLEARN"})
data_X_train, data_Y_train, data_X_val, data_Y_val, data_X_test, data_Y_test = sklearn_linear_regressor_predictor._generate_data_split()
model = sklearn_linear_regressor_predictor.build_model(flatten_default_params(RandomForest.get_default_params()))
trained_model, model_metrics = sklearn_linear_regressor_predictor.train_model(model, data_X_train, data_Y_train, data_X_val, data_Y_val, data_X_test, data_Y_test)
preds = trained_model.predict(target_df)
print("Error Before Loading: ", metrics[0].compute(data_y_df.values, preds))
print(model_metrics)

# Serialize the Model
serialized_model = sklearn_linear_regressor_predictor.serialize_model(trained_model)
# Save the Model
with open("test_model.onnx", "wb") as f:
    f.write(serialized_model)

# Load the Model
import onnxruntime as rt
sess = rt.InferenceSession("test_model.onnx")

# Assemble the inputs
target_dict = {'float_input': target_df.values.astype("float32")}
# print(features_dict)
pred_onx = sess.run([var.name for var in sess.get_outputs()], target_dict)[0]
print("Error After Loading: ", metrics[0].compute(data_y_df.values, pred_onx))

# Plot the predictions
if output_space_dimensionality == 1:
    plt.scatter(target_df.values, data_y_df.values)
    plt.plot(target_df.values, pred_onx, c="red")
    plt.show()


# import pdb; pdb.set_trace()