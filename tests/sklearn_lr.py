# Importing primary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
num_points = 50
input_space_dimensionality = 1
data_y = np.linspace(0, 1, num_points)
data_y_df = pd.DataFrame(data_y)

# Generate Input space features
noise = 0
features_dict = {f"feature_{val}": 2 * data_y + np.random.normal(0, noise, num_points) for val in range(input_space_dimensionality)}

# Convert features Dictionary into dataframe
features_df = pd.DataFrame(features_dict)


# Linear Regression model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(features_df, data_y_df)

# Predicting the results for test set
# y_lm_pred = linear_model.predict(x_test)
# Visualizing Linear Regression
plt.title("Experience vs Salary", fontsize=7)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.annotate('Train points', xy=(6.12337, 60668.89341), arrowprops=dict(arrowstyle='-|>',
                                                                        connectionstyle='angle,angleA=0,angleB=90'), xytext=(2, 125000))
plt.annotate('True Points', xy=(11.75568, 123105.24501), arrowprops=dict(arrowstyle='-|>',
                                                                         connectionstyle='angle,angleA=0,angleB=90'), xytext=(4.5, 175000))
plt.annotate('Prediction points', xy=(14.852, 141118.872), arrowprops=dict(arrowstyle='-|>',
                                                                           connectionstyle='angle,angleA=0,angleB=90'), xytext=(8, 50200))
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.scatter(features_df.values, data_y_df.values, color='#FF8C00', marker='.')
# plt.scatter(x_test, y_test, color='#00FF00', marker='.')
# plt.scatter(x_test, y_lm_pred, color='#00BFFF', marker='.')
plt.plot(features_df.values,
        linear_model.predict(features_df), c='black',  alpha=0.1)
plt.show()
