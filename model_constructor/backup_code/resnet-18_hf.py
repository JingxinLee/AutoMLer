
# 1. Import the necessary libraries and modules
import openml
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from PIL import Image
import numpy as np

# 2. Load the dataset using openml
dataset = openml.datasets.get_dataset('CIFAR_10')

# 3. Get the data from the dataset
X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Convert the data to CSV files for easy reading by the Hugging Face datasets library
# 5.1 Create a pandas DataFrame train_df
train_df = pd.DataFrame(X_train)
train_df['target'] = y_train

# 5.2 Create a pandas DataFrame test_df
test_df = pd.DataFrame(X_test)
test_df['target'] = y_test

# 5.3 Use to_csv function to generate the csv files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# 5.4 Load the csv files with the load_dataset function
train_dataset = load_dataset('csv', data_files='train.csv', split='train')
test_dataset = load_dataset('csv', data_files='test.csv', split='test')

# 6. Initialize the model
model = AutoModelForImageClassification.from_pretrained('microsoft/resnet-18', num_labels=10)

# 7. Preprocess the data for the model
def preprocess_images(examples):
    images = [Image.fromarray(np.array(image, dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)) for image in examples['X_head']]
    examples['pixel_values'] = [np.array(image) for image in images]
    examples['labels'] = examples['target']
    return examples

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
