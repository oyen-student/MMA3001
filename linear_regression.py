"""Simple baselines for hull-resistance regression."""
import numpy as np
from sklearn.linear_model import LinearRegression


class MeanBaseline:
    """Predict the training-set mean for every sample."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
