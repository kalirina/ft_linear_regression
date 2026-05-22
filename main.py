import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
	def __init__(self):
		self.theta0 = 0 # intercept
		self.theta1 = 0 # slope
		self.learning_rate = 0.01
	def learning(self,X,y): # X input, y expected output
		m = len(X) # how many examples we have
		for _ in range(100):
			somme0 = 0
			somme1 = 0
			for i in range(m):
				prediction = self.predict(X[i])
				error = prediction - y[i]
				somme0 += error
				somme1 += error * X[i]
			tmp_theta0 = self.learning_rate * (somme0 / m)
			tmp_theta1 = self.learning_rate * (somme1 / m)
			self.theta0 = self.theta0 - tmp_theta0
			self.theta1 = self.theta1 - tmp_theta1
	def predict(self, X): # regression line
		return self.theta0 + (self.theta1 * X)

def train_model(mileage):
	file = pd.read_csv("data.csv")
	km = file['km']
	price = file['price']
	plt.scatter(km,price, color='blue', label='Data Points')
	plt.title("Mileage vs Car Price")
	plt.xlabel("Mileage (km)")
	plt.ylabel("Price (euro)")
	plt.grid(True)

	normalized_km = km / km.max()
	normalized_price = price / price.max()
	machine = SimpleLinearRegression()
	machine.learning(normalized_km,normalized_price)
	regression_line = machine.predict(normalized_km) * price.max()
	plt.plot(km, regression_line, color='red', label='Regression Line')
	plt.show()

	mileage = float(mileage)
	normalized_mileage = mileage / km.max()
	prediction = machine.predict(normalized_mileage)
	prediction = prediction * price.max()
	return prediction

mileage = input("Enter mileage: ")
res = train_model(mileage)
print("Estimated price: ",res)
