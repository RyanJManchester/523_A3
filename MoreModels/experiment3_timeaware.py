
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from capymoa.models import FIMTDDRegressor
from capymoa.stream import DataStream

# Load dataset (pre-aggregated with 30-min average target)
df = pd.read_csv('MoreModels/loader_f30avg.csv')

# Feature engineering: extract hour of day and station_id as categorical features
df['datetime'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['datetime'].dt.hour
df['station'] = df['station_id'].astype(str)  # Ensure it's string for OneHotEncoder

# Define features and target
features = df[['hour', 'station']]
target = df['target']

# One-hot encode station and hour
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(features)

# Prepare data for stream learning
X_array = X_encoded.astype(np.float32)
y_array = target.to_numpy().astype(np.float32)

# Create stream
stream = DataStream(X_array, y_array)

# Initialize model
model = FIMTDDRegressor()

# Simulate delayed labelling (6 steps)
true_values = []
predictions = []
buffer_X = []
buffer_y = []

for x_i, y_i in stream:
    # Predict first if we have enough history
    if len(buffer_X) >= 6:
        pred = model.predict([buffer_X.pop(0)])[0]
        true = buffer_y.pop(0)
        predictions.append(pred)
        true_values.append(true)

    # Train on current instance
    model.partial_fit([x_i], [y_i])

    # Store instance in buffer
    buffer_X.append(x_i)
    buffer_y.append(y_i)

# Final evaluation
mae = mean_absolute_error(true_values, predictions)
rmse = sqrt(mean_squared_error(true_values, predictions))

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
