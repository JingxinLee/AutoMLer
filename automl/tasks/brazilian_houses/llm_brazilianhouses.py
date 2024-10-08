# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # 加载diabetes数据集
# diabetes = load_diabetes()
# X, y = diabetes.data, diabetes.target

# # 将数据集分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 初始化线性回归模型
# model = LinearRegression()

# # 训练模型
# model.fit(X_train, y_train)

# # 在测试集上进行预测
# y_pred = model.predict(X_test)

# # 评估模型性能
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)  # 2900.1936


import openml
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np

# 加载brazilianhouses数据集
brazilianhouses = openml.datasets.get_dataset(44047)

X, y, _, _ = brazilianhouses.get_data(target=brazilianhouses.default_target_attribute)
# le = LabelEncoder()
# y = le.fit_transform(y)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = pd.DataFrame(X_train)
train_df['target'] = y_train
test_df = pd.DataFrame(X_test)
test_df['target'] = y_test

train_df.to_csv('brazilianhouses_train.csv', index=False)
test_df.to_csv('brazilianhouses_test.csv', index=False)

# 选择最佳模型（这里示例选择随机森林）
# best_model = RandomForestRegressor(n_estimators=100, random_state=42)
# best_model = LinearRegression()
best_model = SVR(kernel='linear')

# 使用交叉验证评估模型性能
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = (-cv_scores) ** 0.5
mean_cv_rmse = cv_rmse_scores.mean()

print("Mean Cross-Validation RMSE:", mean_cv_rmse)

# 训练模型
best_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = best_model.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#Mean Cross-Validation RMSE: 0.39990286079285214
# Mean Squared Error: 0.17119610389610387


# Random Forest Regressor
# Mean Cross-Validation RMSE: 0.06669504684702332
# Mean Squared Error: 0.002429252034859409

# Linear Regression

# SVR
