import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Salary_Data.csv")

# Display basic info
print("Columns:", data.columns)
print(data.head())

# Data Cleaning
data = data.dropna()
data = data.drop_duplicates()

# Convert categorical columns automatically
data = pd.get_dummies(data)

# Split features and target
X = data.drop('Salary', axis=1)
y = data['Salary']

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling (Improves performance)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# Sample Prediction
sample = X_test[0:1]
prediction = model.predict(sample)

print("\n--- Sample Prediction ---")
print("Actual Salary:", y_test.iloc[0])
print("Predicted Salary:", prediction[0])

# Visualization
plt.scatter(y_test, y_pred)

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()