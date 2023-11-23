import os
from openai import OpenAI
from langchain.document_loaders import UnstructuredMarkdownLoader
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import numpy as np

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


def main():
    path = "/Users/jingxinli/Desktop/Study/顶会/Code/model_constructor/MarkdownFiles"  # 这可以是文件或文件夹的路径
    modality = infer_modality(path)
    print(f"The modality of '{path}' is: {modality}")

    dataloader_prompt = f"""
        Your task is to generate a dataloader code snippet for the following modality data. The modality is enclosed in the triple backticks.
        You need to give a Huggingface dataloader code snippet that can load the data into a PyTorch/Tensorflow dataset.
        you should follow these steps when you write the code snippet:
        1. Import the necessary libraries,such as datasets, torchvision, torchaudio,transformers, etc.
        2. Use 'load_dataset' function to load the dataset.
        3. Use 'map' function to preprocess the dataset. Transform the data into the format that the model can understand.
        4. Split the dataset at least into 'train' and 'test' sets.
        5. Use 'DataLoader' function to create a dataloader for the dataset.
        
        At the end, please return the dataloader code snippet with some usage code snippet.
        modality: ```{modality}```
        
    """
    dataloader_response = get_completion(dataloader_prompt)
    print(dataloader_response)


if __name__ == "__main__":
    main()
