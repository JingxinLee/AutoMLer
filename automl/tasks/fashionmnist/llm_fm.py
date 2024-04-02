from datasets import load_dataset
from transformers import (
    ViTForImageClassification,
    ViTFeatureExtractor,
    TrainingArguments,
    Trainer,
)
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import Dataset, DataLoader

# Load the dataset
# dataset = load_dataset(
#     "fashion_mnist",
#     cache_dir="/home/ddp/.cache/huggingface/datasets/fashion_mnist/fashion_mnist/1.0.0/8bbdd6c75ac5dede8443382cce26a0dcd58ea532",
# )

dataset = load_dataset(
    path="/home/ddp/.cache/huggingface/datasets/fashion_mnist/fashion_mnist/1.0.0/8bbdd6c75ac5dede8443382cce26a0dcd58ea532"
)


# Since the dataset might not directly contain an 'image' key, we need to adjust how we process it.
# Define a custom dataset class
class FashionMNISTDataset(Dataset):
    def __init__(self, dataset, feature_extractor):
        self.dataset = dataset
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Dataset provides PIL images and labels
        item = self.dataset[idx]
        image, label = item["image"], item["label"]

        # Convert grayscale image to RGB
        image = image.convert("RGB")

        # Apply feature extractor transformations
        return self.feature_extractor(images=image, return_tensors="pt"), {
            "label": torch.tensor(label)
        }


# Initialize feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

# Prepare datasets
train_dataset = FashionMNISTDataset(dataset["train"], feature_extractor)
eval_dataset = FashionMNISTDataset(dataset["test"], feature_extractor)

# Define the model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", num_labels=10
)

def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='micro')
    return {"accuracy": acc, 'f1': f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    evaluation_strategy="epoch",
)


# Adapt the Trainer class to handle the dataset
class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.args.train_batch_size)

    def get_eval_dataloader(self, eval_dataset):
        return DataLoader(eval_dataset, batch_size=self.args.eval_batch_size)


# Initialize the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=lambda p: {
        "accuracy": (np.argmax(p.predictions, axis=1) == p.label_ids).mean()
    },
)

# Train and evaluate
trainer.train()

eval_results = trainer.evaluate()
print(eval_results)
