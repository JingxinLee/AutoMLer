import autokeras as ak
import sklearn.datasets
import numpy as np

my_trials = 8
my_epochs = 8

# classes = None
classes = ["sci.crypt", "sci.electronics", "sci.med", "sci.space"]
text_twenty = sklearn.datasets.fetch_20newsgroups(categories=classes)

split = int(0.1 * len(text_twenty["data"]))

x_train = text_twenty["data"][: -2 * split]
y_train = text_twenty["target"][: -2 * split]

x_val = text_twenty["data"][-split:]
y_val = text_twenty["target"][-split:]

x_test = text_twenty["data"][-2 * split : -split]
y_test = text_twenty["target"][-2 * split : -split]

print("Target names: ", text_twenty.target_names, "\n")
print(text_twenty["data"][10])


text_classifier = ak.TextClassifier(
    max_trials=my_trials, seed=42, tuner="random"
)

history = text_classifier.fit(
    np.array(x_train),
    y_train,
    validation_data=(np.array(x_val), y_val),
    epochs=my_epochs,
)

predict_val = text_classifier.predict(np.array(x_val))
predict_test = text_classifier.predict(np.array(x_val))

val_accuracy = np.mean(np.array(predict_val.squeeze(), dtype=int) == y_val)
test_accuracy = np.mean(np.array(predict_test.squeeze(), dtype=int) == y_test)

print(f"validation accuracy: {val_accuracy:4f}")
print(f"test set accuracy: {test_accuracy:4f}")
