import json
import requests
import io
import base64
from PIL import Image


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
    


# if __name__ == "__main__":
    # text2image(prompt='Draw a picture of a cat', steps=5)