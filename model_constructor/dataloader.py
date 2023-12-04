import os
from openai import OpenAI
from langchain.document_loaders import UnstructuredMarkdownLoader
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import numpy as np
from torchvision import transforms
from transformers import AutoTokenizer
import torchvision.transforms as T
import augly.text as textaugs

from processfiles import infer_data_modality, infer_folder_modality, infer_modality


_ = load_dotenv(find_dotenv())  # read local .env file
client = OpenAI()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")


# model = "gpt-3.5-turbo-1106" or "gpt-4-1106-preview"
def get_completion(prompt, model="gpt-3.5-turbo-1106"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    return response.choices[0].message.content

# Preprocess 1. PandaGPT -> Text



# Preprocess 2. Stable Diffusion -> Picture
# Statble Diffusion API： bash webui.sh -f --api
def text2image(prompt, steps):
    url = "http://127.0.0.1:7860"
    payload = {
    "prompt": prompt,
    "steps": steps
    }
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()
    image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
    image.save(f"{prompt}.png")
    

# Normalize 

def normalize(modality):
    def normalize_text():
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
        return tokenize_function
    
    def normalize_image(examples):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图像大小
            transforms.ToTensor(),          # 将图像转换为 PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
            ])
        # images = [transform(Image.open(io.BytesIO(image))) for image in examples['image']]
        # return {'image': images}
        return transform
        
    def normalize_audio():
        pass
    
    def normalize_video():
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((256, 256)),  # 调整帧的大小
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
            ])
        
    if modality == "text":
        return normalize_text
    if modality == "image":
        normalize_image()
    if modality == "audio":
        normalize_audio()
    if modality == "video":
        normalize_video()


# Augmentator

def Augmentator(modality):
    def augment_text(text):
        aug_text = textaugs.simulate_typos(input_text)
        return aug_text 
    
    if modality == 'text':
        return augment_text(input_text)

# Define input text
input_text = "Hello, world! How are you today?"
input_text

def main():
    path = "/Users/jingxinli/Desktop/Study/顶会/Code/model_constructor/MarkdownFiles"  # 这可以是文件或文件夹的路径
    modality = infer_modality(path)
    print(f"The modality of '{path}' is: {modality}")
    
    # 1. Preprocess 
    text2image_prompt = "" # What should I say
    text2image(prompt=text2image_prompt, steps=5)
    
    # 2. Normalize
    normalize_func = normalize(modality)
    
    # 3. Augmentator 
    Augmentator = 

    # 4. DataLoader
    dataloader_prompt = f"""
        Your task is to generate a dataloader code snippet for the following modality data. The modality and file path is enclosed in the triple backticks.
        You need to give a Huggingface dataloader code snippet that can load the data into a PyTorch/Tensorflow dataset.
        you should follow these steps when you write the code snippet:
        1. Import the necessary libraries,such as datasets, torchvision, torchaudio,transformers, etc.
        2. Use 'load_dataset' function to load the dataset using the file path I gave you.
        3. Use 'map' function to preprocess the dataset. Transform the data into the format that the model can understand.
        4. Split the dataset at least into 'train' and 'test' sets.
        5. Use 'DataLoader' function to create a dataloader for the dataset.
        
        At the end, please return the dataloader code snippet with some usage code snippet.
        modality: ```{modality}```
        file path: ```{path}```
        
    """
    dataloader_response = get_completion(dataloader_prompt)
    print(dataloader_response)
    augment_text = Augmentator(modality='text')
    print(augment_text)


if __name__ == "__main__":
    main()
