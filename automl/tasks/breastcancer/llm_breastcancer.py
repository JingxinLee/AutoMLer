from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rfc
import xgboost as xgb

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

############# Random Forest ##################
# # Initialize the Random Forest Classifier
# random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# # Train the model
# random_forest_model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = random_forest_model.predict(X_test)



################### SVM #######################
# svc_object = SVC(kernel='poly', degree=8) 
# svc_object.fit(X_train, y_train)
# y_pred = svc_object.predict(X_test) 

# print(classification_report(test_labels, predicted_labels)) 
# print(confusion_matrix(test_labels, predicted_labels)) 
# print(accuracy_score(test_labels, predicted_labels)) 


########### Logistic Regression ###########
# lr_object = LogisticRegression() 
# lr_object.fit(X_train, y_train)
# y_pred = lr_object.predict(X_test)  

# # The following script evaluates the linear regression model:
# print(classification_report(test_labels, predicted_labels)) 
# print(confusion_matrix(test_labels, predicted_labels)) 
# print(accuracy_score(test_labels, predicted_labels)) 


############## XGB ####################
model = xgb.XGBClassifier(use_label_encoder=True, eval_metric='logloss')
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# # Train the model
model.fit(X_train, y_train)

# # 9. Make predictions on the testing set
y_pred = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


########## Random FOrest #######
# Accuracy: 0.9649122807017544
# Confusion Matrix:
# [[40  3]
#  [ 1 70]]
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.98      0.93      0.95        43
#            1       0.96      0.99      0.97        71

#     accuracy                           0.96       114
#    macro avg       0.97      0.96      0.96       114
# weighted avg       0.97      0.96      0.96       114



############# SVC ###############
# Accuracy: 0.9385964912280702
# Confusion Matrix:
# [[36  7]
#  [ 0 71]]
# Classification Report:
#               precision    recall  f1-score   support

#            0       1.00      0.84      0.91        43
#            1       0.91      1.00      0.95        71

#     accuracy                           0.94       114
#    macro avg       0.96      0.92      0.93       114
# weighted avg       0.94      0.94      0.94       114


###### Logistic Regression ########
# Accuracy: 0.9649122807017544
# Confusion Matrix:
# [[40  3]
#  [ 1 70]]
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.98      0.93      0.95        43
#            1       0.96      0.99      0.97        71

#     accuracy                           0.96       114
#    macro avg       0.97      0.96      0.96       114
# weighted avg       0.97      0.96      0.96       114


####### XGB ########
# Accuracy: 0.956140350877193
# Confusion Matrix:
# [[40  3]
#  [ 2 69]]
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.95      0.93      0.94        43
#            1       0.96      0.97      0.97        71

#     accuracy                           0.96       114
#    macro avg       0.96      0.95      0.95       114
# weighted avg       0.96      0.96      0.96       114