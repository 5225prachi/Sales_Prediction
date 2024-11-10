import pandas as pd

# Load the dataset (assuming the CSV file is named 'coffee_sales.csv')
data = pd.read_csv(r"archive (1)\index.csv")


# Display the first few rows to understand the dataset
print(data.head())

# Check for missing values and data types
print(data.info())

# Convert 'datetime' to datetime format and extract useful features
data['datetime'] = pd.to_datetime(data['datetime'])
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['hour'] = data['datetime'].dt.hour

# Convert categorical columns (like 'cash_type' and 'card') to numeric values
data['cash_type'] = data['cash_type'].astype('category').cat.codes
data['card'] = data['card'].astype('category').cat.codes

# Check for missing values
print(data.isnull().sum())

# Dropping the 'datetime' column if it's not directly needed
data = data.drop(columns=['datetime'])

# Feature set (independent variables)
X = data[['money', 'cash_type', 'card', 'month', 'hour']]

# Target variable (dependent variable)
y = data['money']  # Assuming 'money' is what we want to predict for sales

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Make predictions on the entire dataset (or new data)
data['predicted_sales'] = model.predict(X)

# Display the table with the actual and predicted sales
print(data[['money', 'predicted_sales']].head())

# Save the dataset with predicted sales to a new CSV file
data.to_csv("predicted_sales_data.csv", index=False)

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Pie chart - Distribution of Coffee Types
plt.figure(figsize=(8, 6))
coffee_distribution = data['coffee_name'].value_counts()
plt.pie(coffee_distribution, labels=coffee_distribution.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Coffee Types')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

# 2. Histogram - Distribution of Sales (money)
plt.figure(figsize=(8, 6))
sns.histplot(data['money'], bins=20, kde=True)
plt.title('Distribution of Sales (Money)')
plt.xlabel('Sales (Money)')
plt.ylabel('Frequency')
plt.show()

# 3. Scatter plot - Actual vs Predicted Sales
plt.figure(figsize=(8, 6))
plt.scatter(data['money'], data['predicted_sales'], color='blue', label='Predicted Sales', alpha=0.6)
plt.scatter(data['money'], data['money'], color='red', label='Actual Sales', alpha=0.6)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()


