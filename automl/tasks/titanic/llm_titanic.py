import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch

# Load and preprocess the Titanic dataset
df = pd.read_csv(
    "/home/ddp/nlp/github/paper/mypaper_code/automl/data/titanic/train.csv"
)  # Make sure you have the Titanic dataset as "titanic.csv"
df.fillna("", inplace=True)  # Simple way to handle missing values for this example

# Convert features into a single text column
df["text"] = df.apply(lambda row: " ".join(row.values.astype(str)), axis=1)

# Encode the target variable
le = LabelEncoder()
df["survived"] = le.fit_transform(df["survived"])

# Split the dataset
train_df, eval_df = train_test_split(df, test_size=0.2)

# Use a basic transformer model for sequence classification (you might need to adjust this)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# Tokenize text
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


train_encodings = tokenize_function(train_df.to_dict("list"))
eval_encodings = tokenize_function(eval_df.to_dict("list"))


# Create a torch dataset
class TitanicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TitanicDataset(train_encodings, train_df["survived"].tolist())
eval_dataset = TitanicDataset(eval_encodings, eval_df["survived"].tolist())

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    # evaluate_during_training=True,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train and evaluate
trainer.train()

eval_results = trainer.evaluate()
print(eval_results)
