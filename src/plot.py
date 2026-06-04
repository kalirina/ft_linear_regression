import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


def graphs(km, price, theta0, theta1, loss_history):
    plt.figure(figsize=(10, 6))
    plt.scatter(km, price, color='blue', label='Data Points')
    plt.title("Mileage vs Car Price")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (euro)")
    plt.grid(True)
    X_line = np.linspace(km.min(), km.max(), 2)
    y_line = theta0 + theta1 * X_line
    plt.plot(X_line, y_line, color="red", label="Regression line")
    plt.legend()
    plt.savefig("graphs/graph.png")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Iteration of gradient descent")
    plt.ylabel("Loss (average squared error)")
    plt.title("Loss Curve (Training error curve)")
    plt.ticklabel_format(style='plain', axis='y')
    plt.grid(True)
    plt.legend()
    plt.savefig("graphs/loss.png")


def point_in_graph(x, y):
    try:
        file = pd.read_csv("data/data.csv")
    except FileNotFoundError:
        print("Error: data/data.csv not found.")
        return
    km = file['km']
    price = file['price']

    with open("data/thetas.json", "r") as read_file:
        data = json.load(read_file)

    plt.figure(figsize=(10, 6))
    plt.scatter(km, price, color='blue', label='Data Points')
    plt.title("Mileage vs Car Price")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (euro)")

    X_line = np.linspace(km.min(), km.max(), 2)
    y_line = data["theta0"] + data["theta1"] * X_line
    plt.plot(X_line, y_line, color="red", label="Regression line")

    plt.scatter(x, y, color='green', s=150, marker="x", label='Estimation')
    plt.annotate(f"({x:.0f}, {y:.0f})", (x, y),
                 textcoords="offset points", xytext=(10, 10))

    plt.legend()
    plt.grid(True)
    plt.savefig("graphs/graph.png")
