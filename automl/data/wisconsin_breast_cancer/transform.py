import pandas as pd

data = pd.read_csv("/home/ddp/nlp/github/paper/mypaper_code/automl/data/wisconsin_breast_cancer_dataset_edited.csv")
print(data.describe().T)  # Values need to be normalized before fitting.

# 删除id列
data = data.drop(columns=["id"])

# 修改diagosis列的值，M=1， B=0
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# 保存数据
data[:426].to_csv("/home/ddp/nlp/github/paper/mypaper_code/automl/data/wisconsin_breast_cancer_dataset_train.csv", index=False)
data[426:].to_csv("/home/ddp/nlp/github/paper/mypaper_code/automl/data/wisconsin_breast_cancer_dataset_test.csv", index=False)