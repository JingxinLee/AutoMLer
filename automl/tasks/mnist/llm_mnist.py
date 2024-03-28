from transformers import (
    ViTForImageClassification,
    ViTFeatureExtractor,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
import torch


# Load the MNIST dataset from the 'datasets' library
def load_mnist_dataset():
    dataset = load_dataset("mnist")
    transform = Compose(
        [ToTensor(), Lambda(lambda x: x.repeat(3, 1, 1))]
    )  # Transform data

    # Preprocess the images and labels
    def transform_examples(example):
        """Transforms examples to format suitable for ViT."""
        image = transform(example["image"]).float()
        label = example["label"]
        return {"pixel_values": image, "labels": label}

    transformed_dataset = dataset.with_transform(transform_examples)
    return transformed_dataset


transformed_dataset = load_mnist_dataset()

# Initialize the feature extractor and model for ViT
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", num_labels=10
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    remove_unused_columns=False,  # Important for not removing 'labels' and 'pixel_values'
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=transformed_dataset["train"],
    eval_dataset=transformed_dataset["test"],
    tokenizer=feature_extractor,
)

# Train and evaluate the model
trainer.train()

eval_results = trainer.evaluate()
print(eval_results)
