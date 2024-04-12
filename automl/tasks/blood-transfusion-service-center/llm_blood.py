
# 1. Import the necessary libraries and modules
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd

# 2. Load the dataset
dataset = openml.datasets.get_dataset('1464')

# 3. Get the data from the dataset
X, y, _, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Convert the data to a CSV file for easy reading by the Hugging Face datasets library
# 5.1 Create a pandas DataFrame train_df
train_df = pd.DataFrame(X_train)
train_df['target'] = y_train

# 5.2 Create a pandas DataFrame test_df
test_df = pd.DataFrame(X_test)
test_df['target'] = y_test

# 5.3 Use to_csv function to generate the csv file
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

# 5.4 Load the csv file with the load_dataset function
train_dataset = load_dataset('csv', data_files={'train': 'train_data.csv'})['train']
# Convert train_dataset to pandas DataFrame
train_df = train_dataset.to_pandas()
train_target = train_df['target'].replace({1: 0}).replace({2: 1})
train_features = train_df.drop(columns=['target'])

# Convert test_dataset to pandas DataFrame
test_dataset = load_dataset('csv', data_files={'test': 'test_data.csv'})['test']
test_df = test_dataset.to_pandas()
test_df['target'] = test_df['target'].replace({1: 0}).replace({2: 1})  # Convert 2 to 1 for binary classification
test_target = test_df['target']
test_features = test_df.drop(columns=['target'])

model = xgb.XGBClassifier(use_label_encoder=True, eval_metric='logloss')
dtrain = xgb.DMatrix(train_features, label=train_target)
dtest = xgb.DMatrix(test_features, label=test_target)

# Train the model
model.fit(train_features, train_target)

# 9. Make predictions on the testing set
predictions = model.predict(test_features)

# 10. Evaluate the model
accuracy = accuracy_score(test_target, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%") # 0.74