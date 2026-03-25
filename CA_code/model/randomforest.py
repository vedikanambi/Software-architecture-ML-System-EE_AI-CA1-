# ============================================================
# model/randomforest.py — Random Forest model
# Inherits BaseModel and implements all abstract methods
# ============================================================

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from model.base import BaseModel
from Config import RF_N_ESTIMATORS, RF_CLASS_WEIGHT, RF_RANDOM_STATE


class RandomForest(BaseModel):
    """
    Random Forest classifier implementing the BaseModel interface.

    All model-specific code (parameter tuning, sklearn internals,
    result formatting) is hidden here. The controller and modelling
    layer only ever call train(), predict(), print_results().
    """

    def __init__(self):
        super().__init__()
        self.n_estimators = RF_N_ESTIMATORS
        self.class_weight = RF_CLASS_WEIGHT
        self.random_state = RF_RANDOM_STATE
        self.model = None
        self.predictions = None

    def data_transform(self, data):
        """
        No transformation required for Random Forest — returns data as-is.
        Override in other models if needed (e.g., scaling for SVM).
        """
        return data

    def train(self, data):
        """
        Fit the RandomForestClassifier on training data.

        Args:
            data: Data object with X_train and y_train attributes.
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,  # Use all available CPU cores
        )
        self.model.fit(data.X_train, data.y_train)

    def predict(self, X_test):
        """
        Generate predictions for the test set.

        Args:
            X_test: numpy ndarray of test features.

        Returns:
            numpy ndarray of predicted labels.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        self.predictions = self.model.predict(X_test)
        return self.predictions

    def print_results(self, data):
        """
        Print accuracy and full classification report to stdout.

        Args:
            data: Data object with y_test and classes attributes.
        """
        if self.predictions is None:
            raise RuntimeError("No predictions found. Call predict() first.")

        accuracy = accuracy_score(data.y_test, self.predictions)
        print(f"\n  Accuracy: {accuracy:.2%}  ({int(accuracy * len(data.y_test))}/{len(data.y_test)} correct)")
        print("\n  Classification Report:")
        print(classification_report(
            data.y_test,
            self.predictions,
            zero_division=0
        ))
