from autogluon.tabular import TabularDataset, TabularPredictor

# data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
train_data = TabularDataset(
    "/root/paper/mypaper_code/automl/tasks/white-wine/whitewine_train.csv"
)
print(train_data.head())

label = "target"
print(train_data[label].describe())

# Train
predictor = TabularPredictor(label=label, eval_metric="root_mean_squared_error").fit(train_data)

# Test
test_data = TabularDataset(
    "/root/paper/mypaper_code/automl/tasks/white-wine/whitewine_test.csv"
)

y_pred = predictor.predict(test_data.drop(columns=[label]))
print(y_pred.head())

print(predictor.evaluate(test_data, silent=True))

# root_mean_squared_error': -0.5222329678670935