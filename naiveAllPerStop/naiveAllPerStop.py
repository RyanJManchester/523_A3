from capymoa.regressor import FIMTDD
from capymoa.stream import Schema
from capymoa.instance import RegressionInstance
import numpy as np

class NaiveAllPerStopRegressor:
    def __init__(self, stop_ids, window_size=5):
        '''
        Inits regressor for naive approach

        Args:
            stop_ids (str[]): List of stop ids from file
            window_size (int): Number of lag features for model to predict on
        '''

        # Set window sizes and create
        self.window_size = window_size
        self.window = []
        # Create custom schema for dataset
        self.schema = Schema.from_custom(
            feature_names=[
                f"feature_{f}_t{-t}"
                for t in range(self.window_size)
                for f in range(len(stop_ids))
            ],
            dataset_name="OccupancyPredict",
            target_type="numeric"
        )
        # Using FIMTDD as regressor
        self.regressors = {stop_id: FIMTDD(self.schema) for stop_id in stop_ids}

    def update_window(self, value):
        """
        Updates regressor windows with an instance

        Args:
            value (int): value of instance
        """
        # Sliding window
        if len(self.window) == self.window_size:
            self.window.pop(0)
        self.window.append(value)

    def get_window(self):
        """
        Flatten the rows of data in window into a 1d array
        """
        return [item for sublist in self.window for item in sublist]

    def train(self, stop_id, target, features):
        """
        Trains regressor for stop_id on an instance

        Args:
            stop_id (str): String of stop_id data comes from
            target (float): Target value to predict
            features (array): Data to use to predict the value
        """

        # Update window
        self.update_window(features)

        # If there aren't enough instances to implement lag features can't train
        if len(self.window) < self.window_size:
            return
        
        # Build lag features and train regressor
        x = np.array(self.get_window())
        instance = RegressionInstance.from_array(self.schema, x, target)
        self.regressors[str(int(stop_id))].train(instance)

    def predict(self, stop_id):
        """
        Predicts value

        Args:
            stop_id (str): String of stop_id to predict
        """

        # Can't predict without lag features
        if len(self.window) < self.window_size:
            return 0.0
        
        # Build lag features, predict
        x = np.array(self.get_window())
        instance = RegressionInstance.from_array(self.schema, x, 0.0) # Needs target
        return self.regressors[str(int(stop_id))].predict(instance)