
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
def get_data():
    dataset = openml.datasets.get_dataset(4534)
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
    return X, y

# Split the data
X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to CSV
train_df = pd.DataFrame(X_train)
train_df['target'] = y_train
train_df.to_csv('train.csv', index=False)

test_df = pd.DataFrame(X_test)
test_df['target'] = y_test
test_df.to_csv('test.csv', index=False)

# Initialize the Random Forest model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")  # Accuracy: 0.9647218453188603
