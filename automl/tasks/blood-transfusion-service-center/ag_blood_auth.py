from autogluon.tabular import TabularDataset, TabularPredictor

# data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
train_data = TabularDataset(
    "/root/paper/mypaper_code/automl/tasks/blood-transfusion-service-center/blood_train.csv"
)
print(train_data.head())

label = "target"
print(train_data[label].describe())

# Train
predictor = TabularPredictor(label=label).fit(train_data)

# Test
test_data = TabularDataset(
    "/root/paper/mypaper_code/automl/tasks/blood-transfusion-service-center/blood_test.csv"
)

y_pred = predictor.predict(test_data.drop(columns=[label]))
print(y_pred.head())

print(predictor.evaluate(test_data, silent=True))

# accuracy': 0.7466666666666667