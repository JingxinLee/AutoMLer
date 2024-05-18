
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rfc
import xgboost as xgb

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

########## Random Forest #################
# # Initialize the Random Forest model
# model = RandomForestClassifier()

# # Train the model
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)


################### SVM #######################
# svc_object = SVC(kernel='poly', degree=8) 
# svc_object.fit(X_train, y_train)
# y_pred = svc_object.predict(X_test) 

# print(classification_report(test_labels, predicted_labels)) 
# print(confusion_matrix(test_labels, predicted_labels)) 
# print(accuracy_score(test_labels, predicted_labels)) 


########### Logistic Regression ###########
lr_object = LogisticRegression() 
lr_object.fit(X_train, y_train)
y_pred = lr_object.predict(X_test)  

# # The following script evaluates the linear regression model:
# print(classification_report(test_labels, predicted_labels)) 
# print(confusion_matrix(test_labels, predicted_labels)) 
# print(accuracy_score(test_labels, predicted_labels)) 


############## XGB ####################
# model = xgb.XGBClassifier(use_label_encoder=True, eval_metric='logloss')
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)

# # # # Train the model
# model.fit(X_train, y_train)

# # # # 9. Make predictions on the testing set
# y_pred = model.predict(X_test)


# # Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  



# Random Forest Accuracy: 0.9647218453188603
# SVM : Accuracy: 0.9633649932157394
# LR : 
# XGB: