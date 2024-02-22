import autokeras as ak
import sklearn.datasets
import numpy as np

my_epochs = 8
my_trials = 8

my_x, my_y = sklearn.datasets.load_digits(return_X_y=True)

x_train, y_train = my_x[0:1200].reshape(-1, 8, 8), my_y[0:1200]
x_test, y_test = my_x[1200:1400].reshape(-1, 8, 8), my_y[1200:1400]
x_val, y_val = my_x[1400:].reshape(-1, 8, 8), my_y[1400:]

classifier = ak.ImageClassifier(seed=42, max_trials=my_trials, tuner="bayesian")

history = classifier.fit(x_train, y_train, epochs=my_epochs, validation_data=(x_val, y_val))

# results = np.array(classifier.predict(x_test), dtype=int)
# accuracy = np.mean(results.squeeze() == y_test[0:3])
# print(f"Image classification test accuracy = {accuracy:4f}")

results = classifier.predict(x_test)
accuracy = np.mean(np.array(results[0:3].squeeze(), dtype=int) ==  y_test[0:3])
print(f"accuracy = {accuracy:4f}")