
# Step 1: Import necessary libraries
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import torch
from math import sqrt
import numpy as np
from augly.image import aug_np_wrapper, overlay_emoji

# Step 2: Load the CIFAR_10 dataset
dataset = openml.datasets.get_dataset('CIFAR_10')
X, y, _, attribute_names = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Convert the data to CSV files
train_df = pd.DataFrame(X_train)
train_df['target'] = y_train
test_df = pd.DataFrame(X_test)
test_df['target'] = y_test

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# Step 5: Load the CSV files with the Hugging Face datasets library
data_files = {
    'train': 'train.csv',
    'test': 'test.csv'
}
datasets = load_dataset('csv', data_files=data_files)

# Step 6: Initialize the model
model = AutoModelForImageClassification.from_pretrained('microsoft/resnet-18', num_labels=10, ignore_mismatched_sizes=True)

# Step 7: Preprocess the data
def preprocess(examples):
    images = [torch.Tensor(list(img)).view(3, 32, 32) for img in zip(*(examples['a'+str(i)] for i in range(3072)))]
    augmented_images = [aug_np_wrapper(np.array(img, dtype=np.uint8), overlay_emoji, opacity=0.5, y_pos=0.45) for img in images]
    examples['pixel_values'] = [torch.tensor(img) for img in augmented_images]
    examples['labels'] = examples['target']
    return examples

# Apply the preprocess function to the datasets
datasets = datasets.map(preprocess, batched=True)

# Step 8: Train the model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets['train'],
    eval_dataset=datasets['test'],
)

trainer.train()

# Step 9: Make predictions on the testing set
predictions = trainer.predict(datasets['test'])

# Step 10: Evaluate the model
print(predictions.metrics)
