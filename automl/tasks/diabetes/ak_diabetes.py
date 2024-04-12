import autokeras as ak
import sklearn.datasets
import numpy as np

my_trials = 8
my_epochs = 8

# uncomment for larger dataset (housing price prediction)
# my_x, my_y = sklearn.datasets.fetch_california_housing(return_X_y=True)
my_x, my_y = sklearn.datasets.load_diabetes(return_X_y=True)


def normalize_y(my_y):
    # normalize my_y to have mean 0 and std. dev. 1.0

    mean_y = np.mean(my_y)
    std_dev_y = np.std(my_y)

    return (my_y - mean_y) / std_dev_y, mean_y, std_dev_y


def denormalize_y(my_y, my_mean, my_std_dev):
    # return predicted values to y to the original range

    return my_y * my_std_dev + my_mean


my_y, my_mean, my_std_dev = normalize_y(my_y)

split = int(0.1 * my_x.shape[0])

x_train, y_train = my_x[: -2 * split], my_y[: -2 * split]
x_test, y_test = my_x[-2 * split : -split], my_y[-2 * split : -split]
x_val, y_val = my_x[-split:], my_y[-split:]


regression_model = ak.StructuredDataRegressor(seed=42, max_trials=my_trials)

history = regression_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=my_epochs)

mse_loss = lambda x1, x2: np.mean((x1-x2)**2)

results = regression_model.predict(x_test)

my_mse = mse_loss(y_test, results)
print(f"Test MSE = {my_mse:4f}")