from autogluon.tabular import TabularDataset, TabularPredictor

# data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
train_data = TabularDataset(
    "/home/ddp/nlp/github/paper/mypaper_code/model_constructor/generated_scripts/cifar10_data/train_sample.csv"
)
print(train_data.head())

label = "target"
print(train_data[label].describe())

# Train
predictor = TabularPredictor(label=label).fit(train_data)

# Test
test_data = TabularDataset(
    "/home/ddp/nlp/github/paper/mypaper_code/model_constructor/generated_scripts/cifar10_data/test_sample.csv"
)

y_pred = predictor.predict(test_data.drop(columns=[label]))
print(y_pred.head())

print(predictor.evaluate(test_data, silent=True))
