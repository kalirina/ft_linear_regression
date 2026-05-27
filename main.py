import pandas as pd
from SimpleLinearRegression import SimpleLinearRegression
import matplotlib.pyplot as plt
import numpy as np

def learn_from_data(model: SimpleLinearRegression):
	file = pd.read_csv("data.csv")
	km = file['km']
	price = file['price']

	plt.scatter(km,price, color='blue', label='Data Points')
	plt.title("Mileage vs Car Price")
	plt.xlabel("Mileage (km)")
	plt.ylabel("Price (euro)")
	plt.grid(True)

	model.train(km,price)

	sorted_km = np.sort(km)

	plt.plot(
		sorted_km,
		model.predict(sorted_km / model.max_X) * model.max_y,
		color='red',
		label='Regression Line'
	)
	plt.show()

def main():
	user_answer = input("Type 0 to train the model, 1 to estimate a price : ")
	user_answer = int(user_answer)
	model = SimpleLinearRegression()
	if user_answer == 0:
		learn_from_data(model)
	elif user_answer == 1:
		mileage = input("Enter mileage: ")
		mileage = float(mileage)
		prediction = model.predict(mileage)
		print("Estimated price: ",prediction)
	else:
		raise TypeError("Wrong input")

if __name__ == "__main__":
	main()
