"""shell
pip install autokeras
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

import autokeras as ak

"""
## A Simple Example
The first step is to prepare your data. Here we use the [California housing
dataset](
https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
as an example.
"""


# house_dataset = fetch_california_housing()
# df = pd.DataFrame(
#     np.concatenate((house_dataset.data, house_dataset.target.reshape(-1, 1)), axis=1),
#     columns=house_dataset.feature_names + ["Price"],
# )
# train_size = int(df.shape[0] * 0.9)
# df[:train_size].to_csv(
#     "/home/ddp/nlp/github/paper/mypaper_code/automl/data/california_housing/train.csv",
#     index=False,
# )
# df[train_size:].to_csv(
#     "/home/ddp/nlp/github/paper/mypaper_code/automl/data/california_housing/eval.csv",
#     index=False,
# )
train_file_path = (
    "/root/paper/mypaper_code/automl/tasks/white-wine/whitewine_train.csv"
)
test_file_path = (
    "/root/paper/mypaper_code/automl/tasks/white-wine/whitewine_test.csv"
)

"""
The second step is to run the
[StructuredDataRegressor](/structured_data_regressor).
As a quick demo, we set epochs to 10.
You can also leave the epochs unspecified for an adaptive number of epochs.
"""

# Initialize the structured data regressor.
reg = ak.StructuredDataRegressor(
    overwrite=True, max_trials=3
)  # It tries 3 different models.
# Feed the structured data regressor with training data.
reg.fit(
    # The path to the train.csv file.
    train_file_path,
    # The name of the label column.
    "target",
    epochs=10,
)
# Predict with the best model.
predicted_y = reg.predict(test_file_path)
# Evaluate the best model with testing data.
print(reg.evaluate(test_file_path, "target"))

