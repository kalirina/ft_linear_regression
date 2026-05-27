# ft_linear_regression

A simple implementation of linear regression in Python. This project introduces the basics of machine learning by training a model to predict values using gradient descent.

---

## 📚 Project Overview

The goal of this project is to understand how linear regression works internally without using machine learning libraries such as Scikit-learn.

The program:

* Reads a dataset
* Trains a linear regression model
* Finds the best values for:

  * `theta0` (intercept)
  * `theta1` (slope)
* Predicts values using the trained model
* Displays the precision of the model

This project was developed as part of the curriculum at 42 Paris school

---

## 🧠 What is Linear Regression?

Linear regression is a machine learning algorithm used to predict a value based on another value.

The formula of a simple linear regression is:

```text
estimatePrice(mileage) = theta0 + (theta1 * mileage)
```

Where:

* `theta0` = intercept
* `theta1` = slope
* `mileage` = input value
* `estimatePrice` = predicted value

The algorithm improves `theta0` and `theta1` using **gradient descent**.

---

## ⚙️ Features

* CSV dataset parsing
* Data normalization
* Gradient descent implementation
* Prediction program
* Training visualization with graphs
* Error calculation
* Model parameter saving/loading

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/kalirina/ft_linear_regression ft_linear_regression
cd ft_linear_regression
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Train the model

```bash
python training.py
```

This will:

* Read the dataset
* Train the model
* Save the values of `theta0` and `theta1`

---

### 2. Predict a value

```bash
python predict.py
```

Then enter a mileage value:

```text
Enter mileage: 42000
Estimated price: 12500
```

---

## 📊 Gradient Descent

Gradient descent updates the parameters step by step to minimize the prediction error.

Update formulas:

```text
theta0 = theta0 - learningRate * error0
theta1 = theta1 - learningRate * error1
```

The cost decreases over time until the model converges.

---

## 📈 Data Visualization

You can visualize:

* The dataset points
* The regression line
* The evolution of the cost function

Example libraries:

* matplotlib
* numpy
* pandas

---

## 🧪 Example

Dataset:

| Mileage | Price |
| ------- | ----- |
| 10000   | 18000 |
| 30000   | 15000 |
| 50000   | 12000 |

After training:

```text
theta0 = 21000
theta1 = -0.18
```

Prediction:

```text
estimatePrice(40000) = 13800
```

---

## 🛠 Technologies

* Python
* NumPy
* Pandas
* Matplotlib

---

## 📖 Resources

Useful resources:

* Linear Regression
* Gradient Descent
* Mean Squared Error (MSE)
* Normalization

---

## 👤 Author

Made by irkalini at 42 Paris school .
