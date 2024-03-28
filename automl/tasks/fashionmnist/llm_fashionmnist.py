from transformers import (
    ViTForImageClassification,
    ViTFeatureExtractor,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Load Fashion MNIST dataset
# dataset = load_dataset('fashion_mnist', cache_dir='/home/ddp/.cache/huggingface/datasets/fashion_mnist/fashion_mnist/1.0.0/8bbdd6c75ac5dede8443382cce26a0dcd58ea532')
dataset = load_dataset(
    path="/home/ddp/.cache/huggingface/datasets/fashion_mnist/fashion_mnist/1.0.0/8bbdd6c75ac5dede8443382cce26a0dcd58ea532"
)
transform = Compose([ToTensor(), Lambda(lambda x: x.repeat(3, 1, 1))])  # Transform data


def transform_examples(example_batch):
    """Transforms examples to format suitable for ViT."""
    print("example_batch:", example_batch)
    images = [transform(image.convert("RGB")) for image in example_batch["image"]]
    return {"pixel_values": torch.stack(images), "labels": example_batch["label"]}


# Prepare the dataset
train_dataset = dataset["train"].with_transform(transform_examples)
eval_dataset = dataset["test"].with_transform(transform_examples)

# Load feature extractor and model
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
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    remove_unused_columns=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=feature_extractor,
    data_collator=default_data_collator,
)

# Train and evaluate
trainer.train()

eval_results = trainer.evaluate()
print(eval_results)
