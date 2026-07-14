# ===============================
# 1. Import Libraries
# ===============================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ===============================
# 2. Create Dataset
# ===============================

df = pd.DataFrame({
    "Experience": [1,2,3,4,5,6,7,8,9,10],
    "Salary": [30000,35000,45000,50000,60000,65000,70000,80000,85000,95000]
})

print(df)

# ===============================
# 3. Separate Features and Target
# ===============================

X = df[["Experience"]]
y = df["Salary"]

# ===============================
# 4. Train-Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ===============================
# 5. Create Model
# ===============================

model = LinearRegression()

# ===============================
# 6. Train Model
# ===============================

model.fit(X_train, y_train)

# ===============================
# 7. Model Parameters
# ===============================

print("\nSlope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# ===============================
# 8. Predict Test Data
# ===============================

y_pred = model.predict(X_test)

print("\nActual Values")
print(y_test.values)

print("\nPredicted Values")
print(y_pred)

# ===============================
# 9. Evaluate Model
# ===============================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# ===============================
# 10. Predict New Data
# ===============================

new_prediction = model.predict([[12]])

print("\nPredicted Salary for 12 Years Experience:")
print(new_prediction[0])

# ===============================
# 11. Visualization
# ===============================

plt.figure(figsize=(8,5))

plt.scatter(
    X_train,
    y_train,
    color="blue",
    label="Training Data"
)

plt.scatter(
    X_test,
    y_test,
    color="green",
    label="Testing Data"
)

plt.plot(
    X,
    model.predict(X),
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")

plt.legend()

plt.grid(True)

plt.show()