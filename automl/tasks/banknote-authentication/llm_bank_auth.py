import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import openml 
from sklearn.preprocessing import LabelEncoder


# banknote_datadset = pd.read_csv('https://raw.githubusercontent.com/Kuntal-G/Machine-Learning/master/R-machine-learning/data/banknote-authentication.csv')

# # banknote_datadset.head()
# # banknote_datadset.describe()

# dataset_features = banknote_datadset.iloc[:, 0:4].values 
# dataset_labels = banknote_datadset.iloc[:, 4].values 

# 加载banknote_authentication数据集
diabetes = openml.datasets.get_dataset(1462)
X, y, _, _ = diabetes.get_data(target=diabetes.default_target_attribute)
le = LabelEncoder()
y = le.fit_transform(y)

# 将数据集分为训练集和测试集
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

# train_features, test_features, train_labels, test_labels = train_test_split(dataset_features, dataset_labels, test_size=0.2)

############ Random Forest #########################
rfc_object = rfc(n_estimators=200, random_state=0) 
rfc_object.fit(train_features, train_labels) 

predicted_labels = rfc_object.predict(test_features) 

print(classification_report(test_labels, predicted_labels)) 
print(confusion_matrix(test_labels, predicted_labels)) 
print(accuracy_score(test_labels, predicted_labels))  

################### SVM #######################
# svc_object = svc(kernel='poly', degree=8) 
# svc_object.fit(train_features, train_labels)
# predicted_labels = svc_object.predict(test_features) 


########### Logistic Regression ###########
lr_object = LogisticRegression() 
lr_object.fit(train_features, train_labels)
predicted_labels = lr_object.predict(test_features)  

# The following script evaluates the linear regression model:
print(classification_report(test_labels, predicted_labels)) 
print(confusion_matrix(test_labels, predicted_labels)) 
print(accuracy_score(test_labels, predicted_labels)) 

#               precision    recall  f1-score   support

#            0       0.99      1.00      0.99       148
#            1       1.00      0.98      0.99       127

#     accuracy                           0.99       275
#    macro avg       0.99      0.99      0.99       275
# weighted avg       0.99      0.99      0.99       275

# [[148   0]
#  [  2 125]]
# 0.9927272727272727
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99       148
#            1       0.98      0.98      0.98       127

#     accuracy                           0.99       275
#    macro avg       0.99      0.99      0.99       275
# weighted avg       0.99      0.99      0.99       275

# [[146   2]
#  [  2 125]]
# 0.9854545454545455