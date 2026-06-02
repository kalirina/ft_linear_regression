import numpy as np


class SimpleLinearRegression:
    def __init__(self):
        self.theta0 = 0.0  # intercept
        self.theta1 = 0.0  # slope
        self.learning_rate = 0.1
        self.r2 = 0.0
        self.X_line = []
        self.y_line = []
        self.loss_history = []

    def train(self, X, y):  # X input, y expected output
        m = len(X)
        # normalisation
        mean = X.mean(axis=0)
        std = X.std(axis=0)  # standard deviation
        X_norm = (X - mean) / std

        self.loss_history = []
        # gradient descent
        for _ in range(120):
            prediction = self.predict(X_norm)
            error = prediction - y
            # cost function value using MSE
            loss = (1 / (2 * m)) * (error ** 2).sum()
            self.loss_history.append(loss)

            tmp_theta0 = self.learning_rate * (1/m) * error.sum()
            tmp_theta1 = self.learning_rate * (1/m) * (error * X_norm).sum()

            self.theta0 -= tmp_theta0
            self.theta1 -= tmp_theta1
        # denormalisation of thethas
        self.theta0 = self.theta0 - (self.theta1 * mean / std)
        self.theta1 = self.theta1 / std
        print(f"theta0: {self.theta0}, theta1: {self.theta1}")
        # regression line
        self.X_line = np.linspace(X.min(), X.max(), 2)
        self.y_line = self.theta0 + self.theta1 * self.X_line
        # coefficient of determination
        estim_y = self.theta0 + self.theta1 * X
        ss_res = ((y - estim_y) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        self.r2 = 1 - (ss_res / ss_tot)
        print(f"Precision (R**2): {self.r2:.2f}")

    def predict(self, x):
        return self.theta0 + self.theta1 * x
