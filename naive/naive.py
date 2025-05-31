from capymoa.regressor import FIMTDD
from capymoa.stream import Schema
from capymoa.instance import RegressionInstance
import numpy as np

class NaiveRegressor:
    def __init__(self, stop_ids, num_features=5):
        '''
        Inits regressor for naive approach

        Args:
            stop_ids (str[]): List of stop ids from file
            num_features (int): Number of lag features for model to predict on
        '''
        self.num_features = num_features
        # Set window sizes and create
        self.window_size = num_features
        self.windows = {stop_id: [] for stop_id in stop_ids}
        # Create custom schema for dataset
        self.schema = Schema.from_custom(
            feature_names=[str(i) for i in range(num_features)],
            dataset_name="OccupancyPredict",
            target_type="numeric"
        )
        # Using FIMTDD as regressor
        self.regressor = FIMTDD(self.schema)

    def update_window(self, stop_id, value):
        """
        Updates regressor winodws with an instance

        Args:
            stop_id (str): String of stop_id
            value (int): value of instance
        """
        window = self.windows[stop_id]
        # Sliding window
        if len(window) == self.window_size:
            window.pop(0)
        window.append(value)

    def train(self, stop_id, target):
        """
        Trains regressor on an instance

        Args:
            stop_id (str): String of stop_id data comes from
            target (float): Target value to predict
        """
        # Update window
        self.update_window(stop_id, target)

        window = self.windows[stop_id]
        # If there aren't enough instances to implement lag features can't train
        if len(window) < self.num_features:
            return
        
        # Build lag features and train regressor
        x = np.array(window)
        instance = RegressionInstance.from_array(self.schema, x, target)
        self.regressor.train(instance)

    def predict(self, stop_id):
        """
        Predicts value

        Args:
            stop_id (str): String of stop_id to predict from
        """
        window = self.windows[stop_id]
        # Can't predict without lag features
        if len(window) < self.num_features:
            return 0.0
        
        # Build lag features, predict
        x = np.array(window)
        instance = RegressionInstance.from_array(self.schema, x, 0.0) # Needs target
        return self.regressor.predict(instance)


'''
============ Evaluating Model ============
'''

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm # Not needed, shows progress bar

# Load in csv as dataframe
df = pd.read_csv("loader_03-05_2024_30m.csv", index_col=0)

# Get all stop ids
stop_ids = df.columns.to_list()

# Init model
model = NaiveRegressor(stop_ids)

# Keeping track of errors
total_abs_error = 0.0
total_squared_error = 0.0
count = 0

# Window error tracking
error_window = 100
abs_error_buf = 0.0
sqrd_error_buf = 0.0
buf_count = 0
mae_list = []
rmse_list = []

# Loop over values send to regressor as stream
for _, row in tqdm(df.iterrows(), total=len(df)):
    for stop_id in stop_ids:
        value = row[stop_id]

        # Predict and get error
        pred = model.predict(stop_id)
        error = pred - value

        # Update total errors
        total_abs_error += abs(error)
        total_squared_error += error ** 2
        count += 1

        # Update buf errors
        abs_error_buf += abs(error)
        sqrd_error_buf += error ** 2
        buf_count += 1

        # Train on instance
        model.train(stop_id, value)

        # Add windowed error
        if buf_count == error_window:
            mae = abs_error_buf / error_window
            rmse = np.sqrt(sqrd_error_buf / error_window)
            mae_list.append(mae)
            rmse_list.append(rmse)

            # Reset
            abs_error_buf = 0.0
            sqrd_error_buf = 0.0
            buf_count = 0

# Get final metrics
mae = total_abs_error / count
rmse = np.sqrt(total_squared_error / count)

print(f"Final MAE: {mae:.3f}")
print(f"Final RMSE: {rmse:.3f}")

# Plot mae and rmse over windows
plt.figure()
plt.plot(mae_list, label="MAE")
plt.plot(rmse_list, label="RMSE")
plt.title("MAE and RMSE over time")
plt.legend()
plt.grid(True)
plt.show()