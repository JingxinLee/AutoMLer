# To accomplish the task of classifying images into one of 10 categories using the ResNet-50 model on the CIFAR_10 dataset, follow the code snippet below:

# 1. Import necessary libraries and modules
from typing import Optional
import openml
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, AutoConfig,ResNetForImageClassification
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import ToTensor
import os
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 2. Load the CIFAR_10 dataset
dataset = openml.datasets.get_dataset('CIFAR_10')

# 3. Get the data from the dataset
X, y, _, attribute_names = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Convert the data to CSV files and load them
# 5.1 Create a pandas DataFrame for the training data
# train_df = pd.DataFrame(X_train, columns=attribute_names)
# train_df['target'] = y_train

# # 5.2 Create a pandas DataFrame for the testing data
# test_df = pd.DataFrame(X_test, columns=attribute_names)
# test_df['target'] = y_test

# # 5.3 Generate CSV files
# train_df.to_csv('train.csv', index=False)
# test_df.to_csv('test.csv', index=False)

# head -n 1000 train.csv > train_sample.csv 
# head -n 250 test.csv > test_sample.csv
# 5.4 Load the CSV files with the load_dataset function
# train_dataset = load_dataset('csv', data_files='train.csv', split='train')
# test_dataset = load_dataset('csv', data_files='test.csv', split='test')
train_dataset = load_dataset('csv', data_files={'train': 'train_data_sample.csv'}, split="train")
test_dataset = load_dataset('csv', data_files={'test': 'test_data_sample.csv'}, split="test")

# 6. Initialize the ResNet-50 model

model = AutoModelForImageClassification.from_pretrained('microsoft/resnet-50', ignore_mismatched_sizes=True, num_labels=10)

def preprocess_images(examples):
    # 将数据从一维向量转换为图像形状
    images = [torch.Tensor(list(img)).view(3, 32, 32) for img in zip(*(examples[f'a{i}'] for i in range(3072)))]
    examples['pixel_values'] = images
    examples['labels'] = examples['target']
    return examples

# Apply the preprocessing function to the datasets
train_dataset = train_dataset.map(preprocess_images, batched=True)
test_dataset = test_dataset.map(preprocess_images, batched=True)

# 8. Train the model on the train dataset
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

# 9. Make predictions on the testing set
predictions = trainer.predict(test_dataset)

# 10. Evaluate the model
eval_results = trainer.evaluate()

print(eval_results)

