# -*- coding:utf-8 -*-
# @project: PyCharm
# @filename: test.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2025/3/18 22:16
"""

"""
import json
import argparse
import pandas as pd
from vllm import AsyncEngineArgs, AsyncLLMEngine, LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
import re
import time
from tqdm import tqdm


prompt = """**Task Instruction**  
Classify the sentiment of social media text into ONLY "Positive", "Negative", or "Neutral". The text may contain English, French, Arabic, Japanese, Russian or code-mixed content. Output MUST use this exact JSON format: {"result": "sentiment_label"}

**Rules**  
1. Handle all languages and code-mixing  
2. Interpret emojis/slang/sarcasm (e.g., 😡 → Negative, "c'est nul" → Negative)  
3. Strictly follow these case-sensitive labels:  
   - Positive (joy/approval)  
   - Negative (anger/criticism)  
   - Neutral (facts/questions)  

**Output Protocol**  
- Only valid JSON output  
- No explanations or extra fields  
- Escape special characters  
- Examples:  
  Input: "J'adore cette nouvelle fonctionnalité! 😍"  
  Output: {"result": "Positive"}  

  Input: "هذا التطبيق سيء" (Arabic: This app is bad)  
  Output: {"result": "Negative"}  

  Input: "System update v2.1 released"  
  Output: {"result": "Neutral"}  

  Input: "បងតាមខេត្តមានអត់បង"  
  Output: {"result": "Neutral"}  
  
  Input: "おぉーーーーーーーー素晴らしいの一言褒めてますよ😍"  
  Output: {"result": "Positive"}  

**Initialization**   
{{content}}

The final json result is:
  """


def get_json(text):
    pattern = r'```json\s*({.*?})\s*```'

    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_content = match.group(1)  # 提取第一个捕获组
        try:
            json_content = json.loads(json_content)
            return json_content
        except:
            return None
    else:
        return None


def predict(llm, sampling_params, tokenizer, batch_data):
    prompts = []
    for sample in batch_data:
        # print(type(sample))
        content = sample["content"]
        # print(len(content))
        # print(f"content:{content}")
        prompt_user = prompt.replace("{{content}}", content)
        # prompt_user = prompt +  + "\n情感为：\n"
        prompts.append(prompt_user)
    temp_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True) for prompt in prompts]
    prompt_token_ids = tokenizer(temp_prompts).input_ids
    outputs_emotion = llm.generate(prompts=None, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)

    outputs_data = []
    for sample, output_emotion in zip(batch_data, outputs_emotion):
        sentiment = output_emotion.outputs[0].text
        # print(f"sentiment: {sentiment}")
        sample["sentiment"] = sentiment
        outputs_data.append(sample)
    return outputs_data


def get_answer(path, save_path):
    model_path = "/home/public_pretrain_models/Qwen2.5-32B-Instruct-AWQ/"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    stop_words_ids = [151645, 151643]

    llm = LLM(model=model_path, max_model_len=32000, dtype="float16", quantization="awq", tensor_parallel_size=1)
    print("llm")
    sampling_params = SamplingParams(stop_token_ids=stop_words_ids,
                                     top_p=0.8,
                                     temperature=0.9,
                                     repetition_penalty=1.1,
                                     max_tokens=32000,
                                     n=1)

    # df = pd.read_csv(path)
    batch_size = 8
    batch_data = []
    all_data = []

    with open(path, 'r', encoding="utf-8") as f:
        for i, row in enumerate(tqdm(f.readlines())):
            # print(row)
            sample = json.loads(row)
            # print("sample", sample["content"], type(sample))
            batch_data.append(sample)
            if len(batch_data) == batch_size:
                s_time = time.time()
                outputs_data = predict(llm, sampling_params, tokenizer, batch_data)
                e_time = time.time()
                print("time:", e_time - s_time)
                for new_sample in outputs_data:
                    all_data.append(new_sample)
                batch_data = []
        if len(batch_data) != 0:
            outputs_data = predict(llm, sampling_params, tokenizer, batch_data)
            for new_sample in outputs_data:
                all_data.append(new_sample)
    with open(save_path, 'w', encoding="utf-8") as f:
        lines = [json.dumps(line, ensure_ascii=False) for line in all_data]
        f.writelines("\n".join(lines))


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='OriData/math_v1.jsonl', type=str, help='')
    parser.add_argument('--save_path', default="", type=str, help='')
    return parser.parse_args()


def load_sample():
    lines = []
    import random

    for i in range(4):
        with open(f"./data/sample-{i}.jsonl", 'r', encoding="utf-8") as f:
            sample_lines = f.readlines()
            lines += random.choices(sample_lines, k=25)

    with open("./data/sample.jsonl", 'w', encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == '__main__':
    args = set_args()
    # load_sample()
    # # exit()
    # get_answer("./data/sample.jsonl", "./data/sample-result.jsonl")
    # exit()
    print(args.path, args.save_path)
    get_answer(args.path, args.save_path)

# CUDA_VISIBLE_DEVICES=0 nohup python3 -u test.py --path "./data/sample-0.jsonl" --save_path "./data/sample-0-result.jsonl" > ./log0.log 2>&1 &
