# ============================================================
# modelling/data_model.py — Data encapsulation class
# Wraps X_train, X_test, y_train, y_test in one consistent object
# Every ML model receives this same object regardless of algorithm
# ============================================================

import numpy as np
from sklearn.model_selection import train_test_split
from Config import MIN_CLASS_SAMPLES, TEST_SIZE, SPLIT_RANDOM_STATE


class Data:
    """
    Encapsulates all data required for model training and testing.

    Attributes:
        X_train: Training feature matrix (numpy ndarray)
        X_test:  Test feature matrix (numpy ndarray)
        y_train: Training labels (numpy ndarray)
        y_test:  Test labels (numpy ndarray)
        y:       Full label Series before splitting
        classes: Unique class labels after filtering
        embeddings: Reference to full X matrix (for index alignment)
    """

    def __init__(self, X, df, label_col='y2', min_samples=MIN_CLASS_SAMPLES):
        """
        Build a Data object from the embedding matrix and DataFrame.

        Removes classes with fewer than min_samples instances, then
        performs an 80/20 stratified train/test split.

        Args:
            X:          Full TF-IDF embedding matrix (n_samples, n_features)
            df:         DataFrame containing the label column
            label_col:  Name of the column to use as target label
                        (supports chained labels like 'y', or 'y2', 'y3', 'y4')
            min_samples: Minimum instances per class (default: MIN_CLASS_SAMPLES)
        """
        self.embeddings = X
        self.label_col = label_col
        self.min_samples = min_samples

        # Extract labels aligned with X
        labels = df[label_col].values

        # Remove classes with fewer than min_samples instances
        filtered_X, filtered_y = self._filter_rare_classes(X, labels, min_samples)

        if not self.is_valid(filtered_y):
            self.X_train = self.X_test = self.y_train = self.y_test = None
            self.y = None
            self.classes = []
            return

        self.y = filtered_y
        self.classes = np.unique(filtered_y)

        # Stratified train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            filtered_X,
            filtered_y,
            test_size=TEST_SIZE,
            random_state=SPLIT_RANDOM_STATE,
            stratify=filtered_y,
        )

    @staticmethod
    def _filter_rare_classes(X, y, min_samples):
        """
        Remove records belonging to classes with fewer than min_samples instances.

        Args:
            X: Feature matrix
            y: Label array
            min_samples: Threshold

        Returns:
            Filtered X and y arrays.
        """
        unique, counts = np.unique(y, return_counts=True)
        valid_classes = unique[counts >= min_samples]
        mask = np.isin(y, valid_classes)
        return X[mask], y[mask]

    def is_valid(self, y=None):
        """
        Check whether the dataset has enough classes and samples to train.

        A dataset is invalid if:
          - Fewer than 2 unique classes remain after filtering
          - Fewer than 2 * MIN_CLASS_SAMPLES total samples remain

        Args:
            y: Label array to validate (uses self.y if None)

        Returns:
            True if valid, False otherwise.
        """
        labels = y if y is not None else self.y
        if labels is None:
            return False
        unique = np.unique(labels)
        if len(unique) < 2:
            return False
        if len(labels) < 2 * self.min_samples:
            return False
        return True
