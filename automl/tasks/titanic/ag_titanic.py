from autogluon.tabular import TabularDataset, TabularPredictor

# Example Data
# data_url = "https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/"
# train_data = TabularDataset(f"{data_url}train.csv")
train_data = TabularDataset(
    "/home/ddp/nlp/github/paper/mypaper_code/automl/data/titanic/train.csv"
)
print(train_data.head())

label = "survived"
print(train_data[label].describe())


# Training
predictor = TabularPredictor(label=label).fit(train_data)

# Prediction
# test_data = TabularDataset(f'{data_url}test.csv')
test_data = TabularDataset(
    "/home/ddp/nlp/github/paper/mypaper_code/automl/data/titanic/test.csv"
)

y_pred = predictor.predict(test_data.drop(columns=[label]))
print(y_pred.head())

# Evaluation
print(predictor.evaluate(test_data, silent=True))

print(predictor.leaderboard(test_data))
