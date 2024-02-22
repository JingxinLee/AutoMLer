import pandas as pd
import numpy as np
import tensorflow as tf
import autokeras as ak

my_trials = 8
my_epochs = 8

news_df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/News_Final.csv"
)
# news_df = pd.read_csv("/home/ddp/nlp/github/paper/mypaper_code/automl/data/News_Final.csv")


# converting from other formats (such as pandas) to numpy
text_inputs = np.array(news_df.Title + ". " + news_df.Headline).astype("str")
media_success_outputs = news_df.LinkedIn.to_numpy(dtype="int")

# Split the dataset in a train and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    text_inputs, media_success_outputs, test_size=0.2, random_state=10
)

# Initialize the text regressor
reg = ak.TextRegressor(max_trials=my_trials)  # AutoKeras tries different models.

# Callback to avoid overfitting with the EarlyStopping.
cbs = [
    tf.keras.callbacks.EarlyStopping(patience=2),
]

# Search for the best model.
reg.fit(x_train, y_train, epochs=my_epochs, callbacks=cbs)

print(reg.evaluate(x_test, y_test))


# # First we export the model to a keras model
# model = reg.export_model()

# # Now, we ask for the model Sumary:
# print(model.summary())


# ### Predicting some samples
# y_predicted = reg.predict(x_test[0:20])
# for p in list(zip(x_test[0:20], y_test[0:20], [i[0] for i in y_predicted])):
#     print(p)

# #### Improving the model search
# # Callback to avoid overfitting with the EarlyStopping.
# cbs = [tf.keras.callbacks.EarlyStopping(patience=2)]

# input_node = ak.TextInput()
# output_node = ak.TextToIntSequence(max_tokens=20000)(input_node)
# # use ngram as block type
# output_node = ak.TextBlock(block_type='ngram')(input_node)
# # regression output
# output_node = ak.RegressionHead()(output_node)
# # initialize AutoKeras and find the best model
# automodel = ak.AutoModel(inputs=input_node, outputs=output_node,
#                          objective='val_mean_squared_error', max_trials=2)
# automodel.fit(x_train, y_train, callbacks=cbs)


# # Evaluate the custom model with testing data
# automodel.evaluate(x_test, y_test)
