"""Simple baselines for hull-resistance regression."""
import numpy as np
from sklearn.linear_model import LinearRegression


class MeanBaseline:
    """Predict the training-set mean for every sample.

    This baseline contains no feature dependence. It provides a reference for
    whether the input variables improve on predicting a constant target value.
    """

    def fit(self, X, y):
        """Store the mean of the supplied training targets.

        Args:
            X: Feature matrix, accepted for estimator-interface consistency.
            y: One-dimensional training targets.

        Returns:
            This fitted baseline estimator.
        """
        # Estimate the constant from training targets only; another split
        # would leak information into this baseline.
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        """Predict the stored training-target mean for every input row.

        Args:
            X: Feature matrix; its row count determines the number of predictions.

        Returns:
            One-dimensional array of constant predictions.
        """
        return np.full(len(X), self.mean_, dtype=float)


def train_linear_regression(X_train, y_train):
    """Fit an ordinary least-squares regression model.

    Args:
        X_train: Training feature matrix.
        y_train: Training targets in their original units.

    Returns:
        A fitted scikit-learn LinearRegression estimator.
    """
    # Inputs have already been scaled with training-only statistics.
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
