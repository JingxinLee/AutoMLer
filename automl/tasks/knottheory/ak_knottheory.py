"""shell
pip install autokeras
"""

import pandas as pd
import tensorflow as tf

import autokeras as ak

"""
## A Simple Example
The first step is to prepare your data. Here we use the [Titanic
dataset](https://www.kaggle.com/c/titanic) as an example.
"""


# TRAIN_DATA_URL = "/home/ddp/nlp/github/paper/mypaper_code/automl/data/titanic/train.csv"
# TEST_DATA_URL = "/home/ddp/nlp/github/paper/mypaper_code/automl/data/titanic/eval.csv"

# train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
# test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)


train_file_path = (
    "/home/ddp/nlp/github/paper/mypaper_code/automl/data/knot_theory/train.csv"
)
test_file_path = (
    "/home/ddp/nlp/github/paper/mypaper_code/automl/data/knot_theory/test.csv"
)


"""
The second step is to run the
[StructuredDataClassifier](/structured_data_classifier).
As a quick demo, we set epochs to 10.
You can also leave the epochs unspecified for an adaptive number of epochs.
"""

# Initialize the structured data classifier.
clf = ak.StructuredDataClassifier(
    overwrite=True, max_trials=3
)  # It tries 3 different models.

# Feed the structured data classifier with training data.
clf.fit(
    # The path to the train.csv file.
    train_file_path,
    # The name of the label column.
    "signature",
    epochs=10,
)
# Predict with the best model.
predicted_y = clf.predict(test_file_path)
# Evaluate the best model with testing data.
print(clf.evaluate(test_file_path, "signature"))
