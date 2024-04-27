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

# 加载 redwine 数据集
redwine = openml.datasets.get_dataset(44972)
X, y, _, _ = redwine.get_data(target=redwine.default_target_attribute)
# le = LabelEncoder()
# y = le.fit_transform(y)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = pd.DataFrame(X_train)
train_df['target'] = y_train
test_df = pd.DataFrame(X_test)
test_df['target'] = y_test

train_df.to_csv('redwine_train.csv', index=False)
test_df.to_csv('redwine_test.csv', index=False)

# 选择最佳模型（这里示例选择随机森林）
best_model = RandomForestRegressor(n_estimators=100, random_state=42)

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

# Mean Cross-Validation RMSE: 0.6087228664584626
# Mean Squared Error: 0.30123812499999997