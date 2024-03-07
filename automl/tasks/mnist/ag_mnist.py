import tensorflow as tf
from autogluon.tabular import TabularPredictor
import pandas as pd
# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data
# Option 1: Reshape and create DataFrame
train_data = pd.DataFrame(x_train.reshape(-1, 28*28), index=range(len(x_train)))
train_data["label"] = y_train

# Option 2: Iterate and create columns dynamically
# train_data = pd.DataFrame()
# for i in range(len(x_train)):
#     col_name = f"pixel_{i}"
#     train_data[col_name] = x_train[i].flatten()

# train_data["label"] = y_train

# Create test data similarly
# test_data = pd.DataFrame()
# for i in range(len(x_test)):
#     col_name = f"pixel_{i}"
#     test_data[col_name] = x_test[i].flatten()

# test_data["label"] = y_test
test_data = pd.DataFrame(x_test.reshape(-1, 28*28), index=range(len(x_test)))
test_data["label"] = y_test

# Train model
predictor = TabularPredictor(label='label').fit(train_data)

# Evaluate and predict
accuracy = predictor.evaluate(test_data)
predictions = predictor.predict(test_data)

print(f"Accuracy: {accuracy}")
print(f"Predictions: {predictions}")
