# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)
X = np.random.rand(100, 3)  # 100 samples, 3 features
true_weights = [3, 5, 2]
Y = X @ true_weights + 4 + np.random.randn(100)  # Y = 3x1 + 5x2 + 2x3 + 4 + noise

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01, random_state=42)
model.fit(X_train_scaled, Y_train)

Y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Coefficients (weights):", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
```

## Output:
![alt text](<Screenshot 2025-10-06 223957.png>)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
