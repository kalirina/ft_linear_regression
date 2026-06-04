import json
import pandas as pd
from plot import graphs


def train(X, y):  # X input, y expected output
    m = len(X)
    theta0 = 0.0
    theta1 = 0.0
    learning_rate = 0.1
    # normalisation
    mean = X.mean(axis=0)
    std = X.std(axis=0)  # standard deviation
    X_norm = (X - mean) / std

    loss_history = []
    # gradient descent
    for _ in range(150):
        prediction = theta0 + theta1 * X_norm
        error = prediction - y
        # cost function value using MSE
        loss = (1 / (2 * m)) * (error ** 2).sum()
        loss_history.append(loss)

        tmp_theta0 = learning_rate * (1/m) * error.sum()
        tmp_theta1 = learning_rate * (1/m) * (error * X_norm).sum()

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
    # denormalisation of thethas
    theta0 = theta0 - (theta1 * mean / std)
    theta1 = theta1 / std
    print(f"theta0: {theta0}, theta1: {theta1}")
    with open("data/thetas.json", "w") as file:
        json.dump({"theta0": theta0, "theta1": theta1}, file)
    # visualisation
    graphs(X, y, theta0, theta1, loss_history)
    # coefficient of determination
    estim_y = theta0 + theta1 * X
    ss_res = ((y - estim_y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    print(f"Precision (R**2): {r2:.2f}")

def main():
    try:
        file = pd.read_csv("data/data.csv")
    except FileNotFoundError:
        print("Error: data/data.csv not found.")
        return
    km = file['km']
    price = file['price']

    train(km, price)


if __name__ == "__main__":
    main()
