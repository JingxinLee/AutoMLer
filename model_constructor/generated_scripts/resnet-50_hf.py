import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForImageClassification, Trainer
import torch
from torch.utils.data import DataLoader

# Load the dataset
def get_data(dataset_name):
    dataset = openml.datasets.get_dataset(dataset_name)
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
    return X, y

# Split the data
X, y = get_data('4534')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to CSV
train_df = pd.DataFrame(X_train)
train_df['target'] = y_train
train_df.to_csv('train.csv', index=False)

test_df = pd.DataFrame(X_test)
test_df['target'] = y_test
test_df.to_csv('test.csv', index=False)

# Load the CSV files
data_files = {'train': 'train.csv', 'test': 'test.csv'}
dataset = load_dataset('csv', data_files=data_files)

# Initialize the model
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50", ignore_mismatched_sizes=True)

# Define optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + decay * epoch))
optimizers = (optimizer, scheduler)

# Train the model
trainer = Trainer(..., optimizers=optimizers)
trainer.train()

# Make predictions on the testing set
predictions = trainer.predict(test_df)

# Evaluate the model
results = trainer.evaluate()
