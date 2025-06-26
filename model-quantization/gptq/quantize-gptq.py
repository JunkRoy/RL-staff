# -*- coding: utf-8 -*-
# @ Time      : 2025/5/30 16:53
# @ Author    : JunkRoy
# @ e-mail  : shenroy92@gmail.com
# @ Github  : https://github.com/JunkRoy
# @ SoftWare  : PyCharm
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import torch
import logging

device_size = 4
# Specify paths and hyperparameters for quantization
model_path = "your_model_path"
quant_path = "your_quantized_model_path"
quantize_config = BaseQuantizeConfig(
    bits=8,  # 4 or 8
    group_size=128,
    damp_percent=0.01,
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    static_groups=False,
    sym=True,
    true_sequential=True,
    model_name_or_path=None,
    model_file_base_name="model"
)
max_len = 8192

# Load your tokenizer and model with AutoGPTQ
# To learn about loading model to multiple GPUs,
# visit https://github.com/AutoGPTQ/AutoGPTQ/blob/main/docs/tutorial/02-Advanced-Model-Loading-and-Best-Practice.md

tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)

model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config,
    max_memory={i: "40GB" for i in range(device_size)}
)

dataset = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Tell me who you are."},
    {"role": "assistant", "content": "I am a large language model named Qwen..."}
]
data = []
for msg in dataset:
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    model_inputs = tokenizer([text])
    input_ids = torch.tensor(model_inputs.input_ids[:max_len], dtype=torch.int)
    data.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id)))

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
model.quantize(data, cache_examples_on_gpu=False)

model.save_quantized(quant_path, use_safetensors=True)
tokenizer.save_pretrained(quant_path)
