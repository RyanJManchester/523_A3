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