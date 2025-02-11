import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset
df= pd.read_csv(r"C:\Users\merug\OneDrive\Desktop\mahesh\ml\car_price_dataset.csv")

# Step 3: Explore and prepare the data
# Print the first few rows of the dataset to understand its structure
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Select features and target variable
# Assume 'Price' is the target variable and 'Mileage' and 'Year' are the features
X = df[['Mileage', 'Year']]  # You can add more features here based on your dataset
y = df['Price']  # Assuming 'Price' is the target variable

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
# Calculate the Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# Step 8: Visualize the results (Optional)
# Plotting the predicted vs. actual prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices')
plt.show()

# Optional: Coefficients of the model
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
