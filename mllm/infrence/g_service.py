import openai
import gradio as gr
import base64
from io import BytesIO
from PIL import Image

# 设置本地 OpenAI API 代理（vLLM 启动服务的地址）
openai.api_base = "http://127.0.0.1:18400/v1"
openai.api_key = "EMPTY"  # 仅作占位，无需真实 key


# 将图像编码为 base64 并按 OpenAI API 图像格式构造 message
def image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def vqa_with_vllm(image, question):
    base64_img = image_to_base64(image)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": base64_img}}
        ]}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="Qwen2.5-VL-7B-Instruct",
            messages=messages,
            max_tokens=512,
            temperature=0.7
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"出错：{e}"


# Gradio UI
gr.Interface(
    fn=vqa_with_vllm,
    inputs=[
        gr.Image(type="pil", label="上传图片"),
        gr.Textbox(label="请输入问题")
    ],
    outputs=gr.Textbox(label="模型回答"),
    title="Qwen2.5-VL-7B-Instruct 图文问答",
    description="上传图片并输入问题，调用本地 vLLM 服务实现图文问答。"
).launch(server_name="0.0.0.0", server_port=25012, share=True)
