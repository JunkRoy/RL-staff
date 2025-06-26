# -*- coding: UTF-8 -*-
'''
@Project : RL-staff-new 
@File    : train_dpo.py
@Author  : JunkRoy
@Date    : 2025/5/13 22:39
@E-mail  :  shenroy92@gmail.com
@GitHub  : https://github.com/JunkRoy
'''
# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path="/data/public_pretrain_models/Qwen2.5-0.5B-Instruct/"
data_path = "../dataset/ultrafeedback_binarized/"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

train_dataset = load_dataset(data_path, split="train")

training_args = DPOConfig(output_dir="./Qwen2-0.5B-DPO", logging_steps=10)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
