# ============================================================
# model/base.py — Abstract Base Class for all ML models
# Enforces uniform interface: train(), predict(), print_results()
# ============================================================

from abc import ABC, abstractmethod
from utils import string2any


class BaseModel(ABC):
    """
    Abstract base class that every ML model must inherit from.

    Enforces implementation of:
      - train(data): fit the model on training data
      - predict(X_test): generate predictions
      - print_results(data): display classification metrics

    The controller (main.py) only ever calls these three methods —
    it has zero knowledge of each model's internal implementation.
    """

    def __init__(self):
        self.model = None
        self.predictions = None

    @abstractmethod
    def train(self, data):
        """
        Train the model using the Data object.

        Args:
            data: Data instance containing X_train, y_train, etc.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Generate predictions for the test set.

        Args:
            X_test: numpy ndarray of test features.

        Returns:
            numpy ndarray of predicted labels.
        """
        pass

    @abstractmethod
    def print_results(self, data):
        """
        Print classification metrics (accuracy, precision, recall, F1).

        Args:
            data: Data instance containing y_test and class info.
        """
        pass

    @abstractmethod
    def data_transform(self, data):
        """
        Apply any model-specific data transformation before training.

        Args:
            data: Data instance.

        Returns:
            Transformed Data instance.
        """
        pass

    @classmethod
    def build(cls, config: dict):
        """
        Factory method: instantiate a model from a config dictionary.

        Args:
            config: Dict of parameter names → string values.

        Returns:
            Instance of the subclass.
        """
        instance = cls()
        for key, value in config.items():
            if hasattr(instance, key):
                attr_type = type(getattr(instance, key))
                setattr(instance, key, string2any(value, attr_type))
        return instance
