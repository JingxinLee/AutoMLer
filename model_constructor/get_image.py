import json
import requests
import io
import base64
from PIL import Image

url = "http://127.0.0.1:7860"

payload = {
    "prompt": "puppy dog",
    "steps": 5
}

response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
 
r = response.json()
print(r)
image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
image.save('output.png')

# import base64
# import datetime
# import json
# import os

# import requests


# def submit_post(url: str, data: dict):
#     """
#     Submit a POST request to the given URL with the given data.
#     :param url:  url
#     :param data: data
#     :return:  response
#     """
#     return requests.post(url, data=json.dumps(data))


# def save_encoded_image(b64_image: str, output_path: str):
#     """
#     Save the given image to the given output path.
#     :param b64_image:  base64 encoded image
#     :param output_path:  output path
#     :return:  None
#     """
#     # 判断当前目录下是否存在 output 文件夹，如果不存在则创建
#     if not os.path.exists("output"):
#         os.mkdir("output")
#     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#     output_path = f"{output_path}_{timestamp}" + ".png"
#     # 将文件放入当前目录下的 output 文件夹中
#     output_path = f"output/{output_path}"
#     with open(output_path, "wb") as f:
#         f.write(base64.b64decode(b64_image))


# def save_json_file(data: dict, output_path: str):
#     """
#     Save the given data to the given output path.
#     :param data:  data
#     :param output_path:  output path
#     :return:  None
#     """
#     # 忽略 data 中的 images 字段
#     data.pop('images')
#     # 将 data 中的 info 字段转为 json 字符串，info 当前数据需要转义
#     data['info'] = json.loads(data['info'])

#     # 输出 data.info.infotexts
#     info_texts = data['info']['infotexts']
#     for info_text in info_texts:
#         print(info_text)

#     # 判断当前目录下是否存在 output 文件夹，如果不存在则创建
#     if not os.path.exists("output"):
#         os.mkdir("output")
#     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#     output_path = f"{output_path}_{timestamp}" + ".json"
#     # 将文件放入当前目录下的 output 文件夹中
#     output_path = f"output/{output_path}"
#     with open(output_path, "w") as f:
#         json.dump(data, f, indent=4, ensure_ascii=False)


# if __name__ == '__main__':
#     """
#     Example usage: python3 txt2img.py
#     """
#     txt2img_url = "http://127.0.0.1:7860/api/txt2img" # 服务器地址
#     prompt = input("请输入提示词：")
#     negative_prompt = input("请输入反面提示词：")
#     data = {'prompt': prompt, 'negative_prompt': negative_prompt}
#     # 将 data.prompt 中的文本，删除文件名非法字符，已下划线分隔，作为文件名
#     output_path = data['prompt'].replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("\"",
#                                                                                                                   "_").replace(
#         "<", "_").replace(">", "_").replace("|", "_")
#     response = submit_post(txt2img_url, data)
#     print(response)
#     save_encoded_image(response.json()['images'][0], output_path)
#     save_json_file(response.json(), output_path)
