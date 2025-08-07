# -*- coding: utf-8 -*-
# @ Time      : 2025/8/7 14:21
# @ Author    : JunkRoy
# @ e-mail  : shenroy92@gmail.com
# @ Github  : https://github.com/JunkRoy
# @ SoftWare  : PyCharm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import json

# 加载模型和tokenizer
max_seq_length = 2048  # 可以设置更大，但要注意显存
dtype = None  # 让Unsloth自动选择最优的数据类型
load_in_4bit = True  # 使用4bit量化来节省显存

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Chat",  # 你也可以换成其他版本
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 添加LoRA适配器
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA的rank，数值越大模型容量越大，但训练越慢
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,  # Unsloth优化后，通常设为0
    bias="none",  # Unsloth优化后，通常设为none
    use_gradient_checkpointing="unsloth",  # 使用Unsloth的优化
    random_state=3407,
    use_rslora=False,  # 我们支持Rank Stabilized LoRA
    loftq_config=None,  # 以及LoftQ
)

# 设置聊天模板
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",  # 使用Qwen2.5的聊天模板
)


# 数据预处理函数
def formatting_prompts_func(examples):
    convos = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        if input_text:
            text = f"User: {instruction}\n{input_text}\nAssistant: {output}"
        else:
            text = f"User: {instruction}\nAssistant: {output}"
        convos.append(text)
    return {"text": convos}


# 加载和处理数据
with open("your_training_data.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

dataset = Dataset.from_list(train_data)
dataset = dataset.map(formatting_prompts_func, batched=True)

# 设置训练参数
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # 可以设为True来提高训练效率
    args=TrainingArguments(
        per_device_train_batch_size=2,  # 根据你的显存调整
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,  # 通常1-3个epoch就够了
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
        save_steps=100,
        evaluation_strategy="no",  # 如果有验证集可以改为"steps"
    ),
)

# 开始训练
trainer_stats = trainer.train()
