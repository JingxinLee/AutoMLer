# 1. Import necessary libraries and modules
import openml
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 2. Load the dataset using OpenML
dataset = openml.datasets.get_dataset("CIFAR_10")

# 3. Get the data from the dataset
X, y, _, _ = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Convert the data to CSV files
train_df = pd.DataFrame(X_train)
train_df["target"] = y_train
test_df = pd.DataFrame(X_test)
test_df["target"] = y_test

train_csv_path = "train_data_sample.csv"
test_csv_path = "test_data_sample.csv"
# train_df.to_csv(train_csv_path, index=False)
# test_df.to_csv(test_csv_path, index=False)

# 5.4 Load the CSV files with the load_dataset function
data_files = {"train": train_csv_path, "test": test_csv_path}
datasets = load_dataset("csv", data_files=data_files)

# 6. Initialize the model
# model = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=10, ignore_mismatched_sizes=True)
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-18", num_labels=10, ignore_mismatched_sizes=True
)


# 7. Preprocess the data for the model
def preprocess(examples):
    # images = [torch.tensor(list(img)).view(3, 32, 32) for img in zip(*(examples['a'+str(i)] for i in range(3072)))]
    images = [
        torch.tensor(list(img), dtype=torch.float32).view(3, 32, 32)
        # torch.Tensor(list(img)).view(3, 32, 32)
        for img in zip(*(examples[f"a{i}"] for i in range(3072)))
    ]
    examples["pixel_values"] = images
    examples["labels"] = examples["target"]
    return examples


# Apply the preprocess function to the datasets
# datasets['train'] = datasets['train'].map(preprocess, batched=True)
# datasets['test'] = datasets['test'].map(preprocess, batched=True)
datasets = datasets.map(preprocess, batched=True)

# 8. Train the model on the train dataset
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
)

trainer.train()

# 9. Make predictions on the testing set
predictions = trainer.predict(datasets["test"])

# 10. Evaluate the model
eval_results = trainer.evaluate()

print(eval_results)
