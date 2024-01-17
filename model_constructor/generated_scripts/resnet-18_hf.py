# 1. Import necessary libraries and modules
import openml
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import torch

# 2. Load the dataset from OpenML
dataset = openml.datasets.get_dataset("CIFAR_10")

# 3. Get the data from the dataset
X, y, _, _ = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Convert the data to CSV files
# train_df = pd.DataFrame(X_train)
# train_df['target'] = y_train
# test_df = pd.DataFrame(X_test)
# test_df['target'] = y_test

train_csv_path = "train_data_sample.csv"
test_csv_path = "test_data_sample.csv"
# train_df.to_csv(train_csv_path, index=False)
# test_df.to_csv(test_csv_path, index=False)

# 5.4 Load the CSV files with the load_dataset function
data_files = {"train": train_csv_path, "test": test_csv_path}
datasets = load_dataset("csv", data_files=data_files)

# 6. Initialize the model
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-18", ignore_mismatched_sizes=True, num_labels=10
)


# 7. Preprocess the data for the image classification task
def preprocess_data(examples):
    # 7.1 Transform features to 3D (3, 32, 32)
    # images = [torch.tensor(e.values.reshape(3, 32, 32)).float() for e in examples['features']]
    images = [
        torch.Tensor(list(img)).view(3, 32, 32)
        for img in zip(*(examples[f"a{i}"] for i in range(3072)))
    ]
    # 7.2 Save to 'pixel_values'
    examples["pixel_values"] = images
    # 7.3 Save targets to 'labels'
    examples["labels"] = examples["target"]
    return examples


# Apply the preprocess function to the datasets
datasets = datasets.map(preprocess_data, batched=True)

# 8. Train the model on the train dataset
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=64,
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
