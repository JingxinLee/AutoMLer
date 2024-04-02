from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import ViTForImageClassification, AutoTokenizer

model_name = "google/vit-base-patch16-224"  # Example (replace with chosen model)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)  # Adjust class name if needed

# Load Model and Data:
# model_name = "farleyknight-org-username/vit-base-mnist"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = ViTForClassification.from_pretrained(model_name)

# Load MNIST dataset (modify paths as needed)
mnist = load_dataset("mnist")
train_data = mnist["train"]
test_data = mnist["test"]

# Preprocess Data (Tokenization):
def preprocess_function(examples):
  images = examples["image"]
  images = tokenizer(images, padding="max_length", truncation=True)
  return images

train_data = train_data.map(preprocess_function, batched=True)
test_data = test_data.map(preprocess_function, batched=True)



# Train and Evaluate:
# Define training arguments (hyperparameters)
training_args = TrainingArguments(
    output_dir="./results",  # Modify output directory
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=lambda pred: accuracy_metric(pred.predictions, pred.label_ids),
)

# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()
print(eval_result)


# from sklearn.metrics import accuracy_score

# def accuracy_metric(predictions, labels):
#   preds = np.argmax(predictions, axis=1)
#   return {"accuracy": accuracy_score(labels, preds)}
