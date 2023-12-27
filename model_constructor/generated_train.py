# 1st generated snippet
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import load_boston

# # Load the Boston housing dataset
# boston = load_boston()
# X, y = boston.data, boston.target

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the Random Forest Regressor model
# model = RandomForestRegressor(n_estimators=100, random_state=42)

# # Train the model
# model.fit(X_train, y_train)

# # Make predictions on the testing set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)
# ```

# Usage code snippet:
# ```python
# # Usage
# # Create a sample data point for prediction
# sample_data = X_test[0].reshape(1, -1)

# # Make a prediction using the trained model
# predicted_value = model.predict(sample_data)
# print("Predicted value:", predicted_value)

# 2nd generated snippet
import openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 2. Use 'openml.datasets.get_dataset' function to load the dataset
dataset = openml.datasets.get_dataset('boston')

# 3. Use 'get_data' function to get the data from the dataset
X, y, _, _ = dataset.get_data(dataset_format="dataframe", 
                              target=dataset.default_target_attribute
                              )

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize the model
model = LinearRegression()

# 6. Train the model
model.fit(X_train, y_train)

# 7. Make predictions on the testing set
y_pred = model.predict(X_test)

# 8. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
