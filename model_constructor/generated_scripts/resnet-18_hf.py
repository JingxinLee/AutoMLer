
# Step 1: Import necessary libraries and modules
from openml import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import torch
from PIL import Image
import numpy as np

# Step 2: Load the dataset using OpenML
dataset = datasets.get_dataset('CIFAR_10')

# Step 3: Get the data from the dataset
X, y, _, attribute_names = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Convert the data to CSV files and load them
# 5.1 Create a pandas DataFrame for the training data
train_df = pd.DataFrame(X_train, columns=attribute_names)
train_df['target'] = y_train

# 5.2 Create a pandas DataFrame for the testing data
test_df = pd.DataFrame(X_test, columns=attribute_names)
test_df['target'] = y_test

# 5.3 Save the dataframes to CSV files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# 5.4 Load the CSV files with the load_dataset function
train_dataset = load_dataset('csv', data_files='train.csv')['train']
test_dataset = load_dataset('csv', data_files='test.csv')['train']

# Step 6: Initialize the model
model = AutoModelForImageClassification.from_pretrained('microsoft/resnet-18', num_labels=10)

# Step 7: Preprocess the data for the model
def preprocess_images(examples):
    images = [np.array(Image.fromarray(arr.reshape(3, 32, 32).transpose(1, 2, 0)).convert('RGB')) for arr in examples['data']]
    examples['pixel_values'] = [torch.tensor(image).permute(2, 0, 1) for image in images]
    examples['labels'] = examples['target']
    return examples

train_dataset = train_dataset.map(preprocess_images, batched=True)
test_dataset = test_dataset.map(preprocess_images, batched=True)

# Step 8: Train the model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# Step 9: Make predictions on the testing set
predictions = trainer.predict(test_dataset)

# Step 10: Evaluate the model
eval_results = trainer.evaluate()

print(eval_results)
