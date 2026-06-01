import pandas as pd
from SimpleLinearRegression import SimpleLinearRegression
import matplotlib.pyplot as plt
import numpy as np

def graph(km,price,model):
    plt.figure(figsize=(10, 6))
    plt.scatter(km,price, color='blue', label='Data Points')
    plt.title("Mileage vs Car Price")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (euro)")
    plt.grid(True)

    plt.plot(model.X_line, model.y_line, color="red", label="Regression line")
    plt.savefig("graphs/graph.png")

    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_history)
    plt.xlabel("Iteration of gradient descent")
    plt.ylabel("Loss (average squared error)")
    plt.title("Loss Curve (Training error curve)")
    plt.ticklabel_format(style='plain', axis='y')
    plt.grid(True)

    plt.savefig("graphs/loss.png")

def main():
    model = SimpleLinearRegression()

    mileage = input("Enter mileage: ")
    # PARSING
    mileage = float(mileage)

    file = pd.read_csv("data/data.csv")
    km = file['km']
    price = file['price']
    model.train(km,price)
    graph(km,price,model)

    prediction = model.predict(mileage)
    print("Estimated price: ",prediction)

if __name__ == "__main__":
    main()
