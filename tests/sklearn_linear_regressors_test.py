import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate Dummy Data
from framework.metrics.mean_absolute_error import MeanAbsoluteError
from framework.predictors.train.regressors.sklearn.linear_regression import LinearRegression
from utils.predictor import flatten_default_params

num_points = 50
input_space_dimensionality = 1
data_y = np.linspace(0, 1, num_points)
data_y_df = pd.DataFrame(data_y)

# Generate Input space features
noise = 0.2
features_dict = {f"feature_{val}": 2 * data_y + np.random.normal(5, noise, num_points) for val in range(input_space_dimensionality)}

# Convert features Dictionary into dataframe
features_df = pd.DataFrame(features_dict)

# Train the Sklearn Predictor
metrics = [MeanAbsoluteError()]
sklearn_linear_regressor_predictor = LinearRegression(features_df, data_y_df, metrics=metrics, metadata={"name": "test_linear_reg_SKLEARN"})
data_X_train, data_Y_train, data_X_val, data_Y_val, data_X_test, data_Y_test = sklearn_linear_regressor_predictor._generate_data_split()
model = sklearn_linear_regressor_predictor.build_model(flatten_default_params(LinearRegression.get_default_params()))
trained_model, model_metrics = sklearn_linear_regressor_predictor.train_model(model, data_X_train, data_Y_train,
                                                                              data_X_val, data_Y_val,
                                                                              data_X_test, data_Y_test)
preds = trained_model.predict(features_df)
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
features_dict = {'float_input': features_df.values.astype("float32")}
pred_onx = sess.run([var.name for var in sess.get_outputs()], features_dict)[0]
print("Error After Loading: ", metrics[0].compute(data_y_df.values, pred_onx))

# Plot the predictions
plt.scatter(features_df.values, data_y_df.values)
plt.plot(features_df.values, pred_onx)
plt.show()

# import pdb; pdb.set_trace()