import json
import requests
import io
import base64
from PIL import Image
from torchvision import transforms
import os
import torchaudio
from torchvision.io import read_video
from langchain.document_loaders import UnstructuredMarkdownLoader
from collections import Counter


def process_markdown_batch(markdown_files):
    batch_docs = []
    for markdown_file_path in markdown_files:
        markdown_loader = UnstructuredMarkdownLoader(markdown_file_path)
        batch_docs.extend(markdown_loader.load())
    return batch_docs


def iterate_folder_files(root_directory, markdown_files_to_process=[]):
    for root, dirs, files in os.walk(root_directory):
        markdown_files_to_process.extend(
            [os.path.join(root, file) for file in files if file.lower().endswith(".md")]
        )

    return markdown_files_to_process


def process_files_batch(
    process_function,
    markdown_files_to_process=[],
    batch_size=1,
    docs=[],
    processed_files=0,
):
    for i in range(0, len(markdown_files_to_process), batch_size):
        batch = markdown_files_to_process[i : i + batch_size]
        batch_docs = list(map(process_function, [batch]))
        for batch_result in batch_docs:
            docs.extend(batch_result)
            # print(docs)
            processed_files += len(batch)
            # print(f"Processed {processed_files} / {len(markdown_files_to_process)} files")
    return docs


def infer_data_modality(file_path):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    text_extensions = [".txt", ".csv", ".json", ".xml", ".md"]
    audio_extensions = [".wav", ".mp3", ".flac", ".aac"]

    _, ext = os.path.splitext(file_path)
    if ext in image_extensions:
        return "image"
    elif ext in text_extensions:
        return "text"
    elif ext in audio_extensions:
        return "audio"
    else:
        return "unknown"


def infer_folder_modality(folder_path):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    text_extensions = [".txt", ".csv", ".json", ".xml", ".md"]
    audio_extensions = [".wav", ".mp3", ".flac", ".aac"]

    extensions = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            _, ext = os.path.splitext(file)
            extensions.append(ext)

    if extensions:
        most_common_ext, _ = Counter(extensions).most_common(1)[0]
        if most_common_ext in image_extensions:
            return "image"
        elif most_common_ext in text_extensions:
            return "text"
        elif most_common_ext in audio_extensions:
            return "audio"
        else:
            return "unknown"
    else:
        return "empty"


def infer_modality(path):
    if os.path.isfile(path):
        return infer_data_modality(path)
    elif os.path.isdir(path):
        return infer_folder_modality(path)
    else:
        return "invalid"


# 1. PandaGPT -> Text

# from gradio_client import Client
# client = Client("https://gmftby-pandagpt.hf.space/")
# result = client.predict(
# 				"Howdy!",	# str representing input in 'parameter_21' Textbox component
# 				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# str representing input in 'Image' Image component
# 				"https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",	# str representing input in 'Audio' Audio component
# 				"https://github.com/gradio-app/gradio/raw/main/test/test_files/video_sample.mp4",	# str representing input in 'Video' Video component
# 				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# str representing input in 'Thermal Image' Image component
# 				"/root/paper/mypaper_code/model_constructor/TxtFiles",	# str representing input in 'parameter_17' Chatbot component
# 				128,	# int | float representing input in 'Maximum length' Slider component
# 				0.01,	# int | float representing input in 'Top P' Slider component
# 				0.8,	# int | float representing input in 'Temperature' Slider component
# 				fn_index=4
# )
# print(result)


from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
import os
import ipdb
import gradio as gr
import mdtex2html
from model.openllama import OpenLLAMAPEFTModel
import torch
import json

# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}
model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()
print(f'[!] init the 13b model over ...')

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def re_predict(
    input, 
    image_path, 
    audio_path, 
    video_path, 
    thermal_path, 
    chatbot, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    modality_cache, 
):
    # drop the latest query and answers and generate again
    q, a = history.pop()
    chatbot.pop()
    return predict(q, image_path, audio_path, video_path, thermal_path, chatbot, max_length, top_p, temperature, history, modality_cache)


def predict(
    input, 
    image_path, 
    audio_path, 
    video_path, 
    thermal_path, 
    chatbot, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    modality_cache, 
):
    if image_path is None and audio_path is None and video_path is None and thermal_path is None:
        return [(input, "There is no input data provided! Please upload your data and start the conversation.")]
    else:
        print(f'[!] image path: {image_path}\n[!] audio path: {audio_path}\n[!] video path: {video_path}\n[!] thermal path: {thermal_path}')

    # prepare the prompt
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    response = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'video_paths': [video_path] if video_path else [],
        'thermal_paths': [thermal_path] if thermal_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })
    chatbot.append((parse_text(input), parse_text(response)))
    history.append((input, response))
    return chatbot, history, modality_cache




# 2. Stable Diffusion -> Picture
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
def normalize_text(examples):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    return tokenize_function(examples)

def normalize_image(examples):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),          # 将图像转换为 PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
    def apply_transform(examples):
        images = [transform(Image.open(io.BytesIO(image))) for image in examples['image']]
        return {'image': images}
    return apply_transform(examples)
    
def normalize_audio(examples):
    def apply_transform(examples):
        audios = [torchaudio.load(ex['audio'])[0] / torch.max(torch.abs(torchaudio.load(ex['audio'])[0])) for ex in examples]
        return {'audio': audios}
    return apply_transform(examples)


def normalize_video(examples):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # 调整帧的大小
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
    def apply_transform(examples):
        normalized_videos = []
        for video_file in examples['video_file']:
            # 读取视频
            video, _, _ = read_video(video_file)
            # 归一化
            video = video / 255.0
            normalized_video = transform(video)
            normalized_videos.append(normalized_video)

        return {'video': normalized_videos}
    return apply_transform(examples)
  
# # Normalize  with folder
# Text 
def normalize_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # 基本的文本清洗，如转换为小写
    text = text.lower()

    return text

def normalize_text_folder(folder_path):
    normalized_texts = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            normalized_text = normalize_text_file(file_path)
            normalized_texts.append(normalized_text)

    return normalized_texts

# Image
def normalize_image_folder(folder_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),          # 将图像转换为 PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    normalized_images = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查是否为文件
        if os.path.isfile(file_path):
            try:
                # 打开并应用转换
                with open(file_path, 'rb') as file:
                    image = Image.open(io.BytesIO(file.read())).convert('RGB')
                    normalized_images.append(transform(image))
            except IOError:
                print(f"Could not open or read the file {file_path}")

    return normalized_images

# video 
def normalize_video_file(file_path):
    video, _, _ = read_video(file_path)

    # 定义归一化转换
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    video = video / 255.0
    normalized_video = normalize(video)

    return normalized_video

def normalize_video_folder(folder_path):
    normalized_videos = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and file_path.endswith(('.mp4', '.avi')):
            normalized_video = normalize_video_file(file_path)
            normalized_videos.append(normalized_video)

    return normalized_videos

# Audio 
def load_and_normalize_audio(filename):
    # 加载音频文件
    waveform, sample_rate = torchaudio.load(filename)

    # 归一化音频波形
    waveform = waveform / torch.max(torch.abs(waveform))

    return waveform, sample_rate

def normalize_audio_folder(folder_path):
    normalized_audios = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查是否为音频文件
        if os.path.isfile(file_path) and file_path.endswith(('.wav', '.mp3', '.flac')):
            normalized_waveform, sample_rate = load_and_normalize_audio(file_path)
            normalized_audios.append((normalized_waveform, sample_rate))

    return normalized_audios

# if __name__ == "__main__":
    # text2image(prompt='Draw a picture of a cat', steps=5)