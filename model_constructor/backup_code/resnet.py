from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
from torchsummary import summary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").cuda()

inputs = processor(image, return_tensors="pt")
inputs = inputs.to(device)
print(inputs['pixel_values'].shape)

with torch.no_grad():
    logits = model(**inputs).logits
summary(model, (3, 224, 224))

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
