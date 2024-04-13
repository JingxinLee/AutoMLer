
# Step 1: Import necessary libraries
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder

# Step 2: Load the dataset
dataset = openml.datasets.get_dataset('31')
X, y, _, attribute_names = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)

######### Preprocessing #########
features = ['checking_status','credit_history','purpose','savings_status','employment','personal_status','other_parties','property_magnitude','other_payment_plans','housing','job','own_telephone','foreign_worker']
credit= pd.get_dummies(X, columns = features)

le = LabelEncoder()
Y = le.fit_transform(y)


# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Step 5: Convert the data to CSV files for easy reading
train_df = pd.DataFrame(X_train)
train_df['target'] = y_train
test_df = pd.DataFrame(X_test)
test_df['target'] = y_test

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# Step 5.4: Load the CSV files with the Hugging Face datasets library
data_files = {
    'train': 'train.csv',
    'test': 'test.csv'
}
datasets = load_dataset('csv', data_files=data_files)

# Since we are using LightGBM, steps 6 and 7 related to tokenization and image preprocessing are not applicable.

# Step 8: Initialize the LightGBM model
model = lgb.LGBMClassifier()

# Step 8: Define optimizers (Not applicable as LightGBM handles this internally)

# Step 8: Train the model on the train dataset
model.fit(X_train, y_train)

# Step 9: Make predictions on the testing set
predictions = model.predict(X_test)

# Step 10: Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy}") # 0.7900
