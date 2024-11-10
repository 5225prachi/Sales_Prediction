import matplotlib.pyplot as plt

# Plot actual vs predicted sales
plt.scatter(data['money'], data['predicted_sales'], color='blue', label='Predicted Sales')
plt.scatter(data['money'], data['money'], color='red', label='Actual Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()
