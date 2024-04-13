import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import openml 
from sklearn.preprocessing import LabelEncoder
import autokeras as ak
import seaborn as sns


# banknote_datadset = pd.read_csv('https://raw.githubusercontent.com/Kuntal-G/Machine-Learning/master/R-machine-learning/data/banknote-authentication.csv')

# # banknote_datadset.head()
# # banknote_datadset.describe()

# dataset_features = banknote_datadset.iloc[:, 0:4].values 
# dataset_labels = banknote_datadset.iloc[:, 4].values 

# 加载blood数据集
creditg = openml.datasets.get_dataset(31)
X, y, _, _ = creditg.get_data(target=creditg.default_target_attribute)
le = LabelEncoder()
y = le.fit_transform(y)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the search
clf = ak.StructuredDataClassifier(max_trials=10)
clf.fit(X_train, y_train, verbose=1, epochs=10)


# Evaluate the classifier on test data
_, acc = clf.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

# Predicting the Test set results
y_pred = clf.predict(X_test)
y_pred = y_pred > 0.5

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)

# get the final best performing model
model = clf.export_model()
print(model.summary())

# Save the model
model.save("creditg_model.h5")

# Accuracy =  0.7200000286102295 
