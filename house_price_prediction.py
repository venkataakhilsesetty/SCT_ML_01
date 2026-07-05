import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# HOUSE PRICE PREDICTION USING LINEAR REGRESSION
# ==========================================================

# Load Dataset
df = pd.read_csv("Housing.csv")

print("="*60)
print("      HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
print("="*60)

# ==========================================================
# Dataset Information
# ==========================================================

print("\nFirst 5 Records")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ==========================================================
# Feature Selection
# ==========================================================

X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# ==========================================================
# Split Dataset
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# Create and Train Model
# ==========================================================

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================================================
# Prediction on Test Data
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# Regression Equation
# ==========================================================

print("\nRegression Equation")

print(
    "Price = {:.2f} + ({:.2f} × Area) + ({:.2f} × Bedrooms) + ({:.2f} × Bathrooms)"
    .format(
        model.intercept_,
        model.coef_[0],
        model.coef_[1],
        model.coef_[2]
    )
)

# ==========================================================
# Performance Evaluation
# ==========================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-"*40)
print("Mean Absolute Error (MAE) :", round(mae, 2))
print("Mean Squared Error (MSE)  :", round(mse, 2))
print("Root Mean Squared Error   :", round(rmse, 2))
print("R² Score                  :", round(r2, 4))

# ==========================================================
# Actual vs Predicted Values
# ==========================================================

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices")
print(results.head(10))

# ==========================================================
# Predict House Price (User Input)
# ==========================================================

print("\nPredict House Price")
print("-"*40)

area = float(input("Enter Area (sq.ft): "))

# Average values for remaining features
bedrooms = df['bedrooms'].mean()
bathrooms = df['bathrooms'].mean()

new_house = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms]
})

prediction = model.predict(new_house)

print("\nPrediction Result")
print("-"*40)
print("Area :", area, "sq.ft")
print("Predicted House Price = {:.2f}".format(prediction[0]))

# ==========================================================
# Visualization
# ==========================================================

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.grid(True)

plt.show()