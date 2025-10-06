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
# 1. Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Data (X: Hours Studied, Y: Marks Scored)
X = np.array([2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7]).reshape(-1, 1)
Y = np.array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25])

# 2. Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 3. Model Training
model = LinearRegression()
model.fit(X_train, Y_train)

# 4. Making Predictions
Y_pred = model.predict(X_test)

# 5. Model Evaluation
print(f"Intercept (B0): {model.intercept_}")
print(f"Slope (B1): {model.coef_[0]}")
print("---")
print(f"Mean Squared Error (MSE): {mean_squared_error(Y_test, Y_pred):.2f}")
print(f"R-squared (R2) Score: {r2_score(Y_test, Y_pred):.2f}")

# Example prediction: Predict marks for 6.0 hours studied
new_hours = np.array([[6.0]])
predicted_marks = model.predict(new_hours)
print(f"Predicted Marks for 6.0 hours: {predicted_marks[0]:.2f}")

```

## Output:
<img width="604" height="208" alt="Screenshot 2025-10-06 233907" src="https://github.com/user-attachments/assets/da38a21b-ad0b-487b-b652-70ca7e079799" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
