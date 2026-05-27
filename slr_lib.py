import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file = pd.read_csv("data.csv")
km = file[['km']]
price = file[['price']]
model = LinearRegression()
model.fit(km, price)

Y_pred = model.predict(km)

plt.figure(figsize=(8,6))
plt.scatter(km, price, color='blue', label='Data Points')
plt.plot(km, Y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression on Random Dataset')
plt.xlabel('km')
plt.ylabel('price')
plt.legend()
plt.grid(True)
plt.show()
