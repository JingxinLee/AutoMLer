
# 1. Import the necessary libraries and modules
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rfc

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
train_labels = train_df['target'].replace({1: 0}).replace({2: 1})
train_features = train_df.drop(columns=['target'])

# Convert test_dataset to pandas DataFrame
test_dataset = load_dataset('csv', data_files={'test': 'test_data.csv'})['test']
test_df = test_dataset.to_pandas()
test_df['target'] = test_df['target'].replace({1: 0}).replace({2: 1})  # Convert 2 to 1 for binary classification
test_labels = test_df['target']
test_features = test_df.drop(columns=['target'])



############ Random Forest #########################
# rfc_object = rfc(n_estimators=200, random_state=0) 
# rfc_object.fit(train_features, train_labels) 

# predicted_labels = rfc_object.predict(test_features) 

# print(classification_report(test_labels, predicted_labels)) 
# print(confusion_matrix(test_labels, predicted_labels)) 
# print(accuracy_score(test_labels, predicted_labels))  

################### SVM #######################
# svc_object = SVC(kernel='poly', degree=8) 
# svc_object.fit(train_features, train_labels)
# predicted_labels = svc_object.predict(test_features) 

# print(classification_report(test_labels, predicted_labels)) 
# print(confusion_matrix(test_labels, predicted_labels)) 
# print(accuracy_score(test_labels, predicted_labels)) 


########### Logistic Regression ###########
lr_object = LogisticRegression() 
lr_object.fit(train_features, train_labels)
predicted_labels = lr_object.predict(test_features)  

# The following script evaluates the linear regression model:
print(classification_report(test_labels, predicted_labels)) 
print(confusion_matrix(test_labels, predicted_labels)) 
print(accuracy_score(test_labels, predicted_labels)) 


# model = xgb.XGBClassifier(use_label_encoder=True, eval_metric='logloss')
# dtrain = xgb.DMatrix(train_features, label=train_target)
# dtest = xgb.DMatrix(test_features, label=test_target)

# # Train the model
# model.fit(train_features, train_target)

# # 9. Make predictions on the testing set
# predictions = model.predict(test_features)

# 10. Evaluate the model
# accuracy = accuracy_score(test_target, predictions)
# print(f"Accuracy: {accuracy * 100:.2f}%") # 0.74


######## Random Forest ########
#               precision    recall  f1-score   support

#            0       0.78      0.88      0.82       113
#            1       0.39      0.24      0.30        37

#     accuracy                           0.72       150
#    macro avg       0.59      0.56      0.56       150
# weighted avg       0.68      0.72      0.70       150

# [[99 14]
#  [28  9]]
# 0.72


#### SVC  ########
#               precision    recall  f1-score   support

#            0       0.77      0.96      0.85       113
#            1       0.50      0.11      0.18        37

#     accuracy                           0.75       150
#    macro avg       0.63      0.54      0.52       150
# weighted avg       0.70      0.75      0.69       150

# [[109   4]
#  [ 33   4]]
# 0.7533333333333333


#### Logistic Regression ######
#               precision    recall  f1-score   support

#            0       0.77      0.97      0.86       113
#            1       0.57      0.11      0.18        37

#     accuracy                           0.76       150
#    macro avg       0.67      0.54      0.52       150
# weighted avg       0.72      0.76      0.69       150

# [[110   3]
#  [ 33   4]]
# 0.76