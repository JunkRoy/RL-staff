# -*- coding: utf-8 -*-
# @ Time      : 2025/5/28 17:34
# @ Author    : JunkRoy
# @ e-mail  : shenroy92@gmail.com
# @ Github  : https://github.com/JunkRoy
# @ SoftWare  : PyCharm
import base64
import requests

# 配置
API_URL = "http://localhost:18400/v1/chat/completions"
IMAGE_PATH = "./images/2.png"

# 1. 读取图像并编码为 base64
with open(IMAGE_PATH, "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode("utf-8")

# 2. 构造请求 payload
payload = {
    "model": "mllm",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": "抽取文本中的所有公式，注意公式使用Latex表达？"
                }
            ]
        }
    ],
    "temperature": 0.6
}

# 3. 发送请求
response = requests.post(API_URL, json=payload)

# 4. 输出结果
if response.ok:
    print(response.json()["choices"][0]["message"]["content"])
else:
    print("请求失败：", response.status_code, response.text)
