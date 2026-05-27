import numpy as np

class SimpleLinearRegression:
	def __init__(self):
		self.theta0 = 0 # intercept
		self.theta1 = 0 # slope
		self.learning_rate = 0.01
	def train(self,X,y): # X input, y expected output
		m = len(X) # how many examples we have
		# normalisation
		self.max_X = X.max()
		self.max_y = y.max()
		n_X = X / self.max_X
		n_y = y / self.max_y
		# gradient descent
		for _ in range(1000):
			somme0 = 0
			somme1 = 0
			for i in range(m):
				prediction = self.predict(n_X[i])
				error = prediction - n_y[i]
				somme0 += error
				somme1 += error * n_X[i]
			tmp_theta0 = self.learning_rate * (somme0 / m)
			tmp_theta1 = self.learning_rate * (somme1 / m)
			self.theta0 = self.theta0 - tmp_theta0
			self.theta1 = self.theta1 - tmp_theta1
		print("thetha0 : ", self.theta0)
		print("thetha1 : ", self.theta1)
		self.y_pred = self.predict(n_X) * self.max_y
	def predict(self, x): # regression line
			return self.theta0 + self.theta1 * x
