# 1. Import necessary libraries and modules
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import torch
from math import sqrt
import numpy as np
from augly.image import aug_np_wrapper, overlay_emoji

# 2. Load the dataset using OpenML
dataset = openml.datasets.get_dataset('CIFAR_10')

# 3. Get the data from the dataset
X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Convert the data to CSV files
train_df = pd.DataFrame(X_train)
train_df['target'] = y_train
test_df = pd.DataFrame(X_test)
test_df['target'] = y_test

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# 5.4 Load the CSV files with the load_dataset function
data_files = {
    'train': 'train.csv',
    'test': 'test.csv'
}
datasets = load_dataset('csv', data_files=data_files)

# 6. Initialize the model
model = AutoModelForImageClassification.from_pretrained('microsoft/resnet-18', ignore_mismatched_sizes=True)

# 7. Preprocess the data
def preprocess(examples):
    num_features = 3072  # CIFAR-10 images are 32x32x3
    image_side_length = int(sqrt(num_features / 3))
    
    # 7.1 Augment the data (optional)
    augmented_images = [aug_np_wrapper(np.array(img, dtype=np.uint8).reshape((3, image_side_length, image_side_length)), overlay_emoji, opacity=0.5, y_pos=0.45) for img in zip(*(examples["a" + str(i)] for i in range(num_features)))]
    
    # 7.2 Transform features to 3D tensors
    images = [torch.Tensor(img).view(3, image_side_length, image_side_length) for img in augmented_images]
    
    # 7.3 Save to 'pixel_values' and 'labels'
    examples['pixel_values'] = images
    examples['labels'] = [label for label in examples['target']]
    
    return examples

# Apply the preprocess function to the datasets
datasets = datasets.map(preprocess, batched=True)

# 8. Train the model
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
    train_dataset=datasets['train'],
    eval_dataset=datasets['test'],
)

trainer.train()

# 9. Make predictions on the testing set
predictions = trainer.predict(datasets['test'])

# 10. Evaluate the model
eval_results = trainer.evaluate()

print(eval_results)
