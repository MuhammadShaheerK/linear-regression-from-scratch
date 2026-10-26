"""Batch gradient descent with an intercept."""
import numpy as np


class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=5000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.weights_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self.cost_history_ = []
        for _ in range(self.iterations):
            error = X @ self.weights_ + self.intercept_ - y
            self.cost_history_.append(float(np.mean(error ** 2) / 2))
            self.weights_ -= self.learning_rate * (X.T @ error / len(y))
            self.intercept_ -= self.learning_rate * error.mean()
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights_ + self.intercept_
