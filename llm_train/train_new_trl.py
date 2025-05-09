import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments, set_seed
from trl import SFTTrainer, SFTConfig
# from utils import create_and_prepare_model, create_datasets
import json
# from torch.utils.data import Dataset
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
from peft import LoraConfig
import math


def build_qwen2_prompt_dataset(data_path, tokenizer, max_len, max_src_len, is_skip=True):
    nl_tokens = tokenizer.encode("\n")

    def _tokenize_str(role, content):
        return f"{role}\n{content}", tokenizer.encode(role) + nl_tokens + tokenizer.encode(content)

    all_data = []
    skip_data_number = 0
    with open(data_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):

            skip_flag = False
            im_start_tokens = tokenizer.encode("<|im_start|>")
            im_end_tokens = tokenizer.encode("<|im_end|>")
            sys_prompt = "You are a helpful assistant."
            system_text, system_tokens_part = _tokenize_str("system", sys_prompt)
            system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

            sample = json.loads(line.strip())
            sample = [sample]
            input_ids = []
            labels = []

            if len(sample) > 1:
                history = sample[:-1]
                current = sample[-1]
            else:
                history = []
                current = sample[0]

            for i_c, chat in enumerate(history):
                _, query_tokens_part = _tokenize_str("user", chat["instruction"] + chat["input"])
                query_tokens = im_start_tokens + query_tokens_part + im_end_tokens

                _, response_tokens_part = _tokenize_str("assistant", chat["output"])
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens
                input_ids.extend(nl_tokens + query_tokens + nl_tokens + response_tokens)
                labels.extend([-100] * (len(query_tokens) + 1 + len(nl_tokens) * 2) + response_tokens[1:])

            prompt_id = im_start_tokens + _tokenize_str("user", current["instruction"] + current["input"])[
                1] + im_end_tokens

            if len(prompt_id) > max_src_len:
                input_ids = nl_tokens + prompt_id[:max_src_len - 1] + [prompt_id[-1]]
                labels = [-100] * (len(input_ids))
                skip_flag = True
            else:
                input_ids.extend(nl_tokens + prompt_id)
                labels.extend([-100] * (len(prompt_id) + len(nl_tokens)))
                if len(input_ids) > max_src_len:
                    skip_flag = True
                    input_ids = input_ids[-max_src_len:]
                    labels = labels[-max_src_len:]
            # print(len(input_ids), len(labels))
            assert len(input_ids) == len(labels)
            output_id = im_start_tokens + _tokenize_str("assistant", current["output"])[1] + im_end_tokens
            max_tgt_len = max_len - len(input_ids) - len(system_tokens)
            if len(output_id) > max_tgt_len:
                output_id = output_id[:max_tgt_len - 1] + [output_id[-1]]
                skip_flag = True

            input_ids = system_tokens + input_ids + nl_tokens + output_id
            labels = [-100] * len(system_tokens) + labels + [-100] * (1 + len(nl_tokens)) + output_id[1:]

            assert len(input_ids) == len(labels)
            assert len(input_ids) <= max_len + 10
            if is_skip and skip_flag:
                skip_data_number += 1
                continue
            all_data.append({"input_ids": input_ids, "labels": labels})
    print(
        "the number of skipping data is {}, the proportion is {}".format(skip_data_number, skip_data_number / (
                len(all_data) + skip_data_number)))

    return Dataset.from_list(all_data)


class DataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        print(f"batch: {batch}")
        lengths = [len(instance["input_ids"]) for instance in batch]

        # batch_max_len = max(max(lengths), 4608)
        batch_max_len = max(lengths)
        # batch_max_len = math.ceil(max(lengths) / 8) * 8
        # print(batch_max_len)
        input_ids_batch, labels_batch = [], []
        for instance in batch:
            input_ids = instance["input_ids"]
            labels = instance["labels"]

            padding_len = batch_max_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_len
            labels = labels + [-100] * padding_len

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)

        return {"input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
                "labels": torch.tensor(labels_batch, dtype=torch.long)}


# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="float32",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )


@dataclass
class DataTrainingArguments:
    train_path: Optional[str] = field(
        default="timdettmers/openassistant-guanaco"
    )
    test_path: Optional[str] = field(
        default="timdettmers/openassistant"
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    max_seq_length: Optional[int] = field(default=512)
    max_src_length: Optional[int] = field(default=256)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # model
    # model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

    compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
    quant_storage_dtype = getattr(torch, model_args.bnb_4bit_quant_storage_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_args.use_4bit_quantization,
        bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=model_args.use_nested_quant,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

    if compute_dtype == torch.float16 and model_args.use_4bit_quantization:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)
    elif model_args.use_8bit_quantization:
        bnb_config = BitsAndBytesConfig(load_in_8bit=model_args.use_8bit_quantization)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                 quantization_config=bnb_config,
                                                 trust_remote_code=True,
                                                 attn_implementation="flash_attention_2" if model_args.use_flash_attn else "eager",
                                                 torch_dtype=quant_storage_dtype or torch.float32)

    if model_args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=model_args.lora_target_modules.split(",")
            if model_args.lora_target_modules != "all-linear"
            else model_args.lora_target_modules,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    # datasets
    # train_dataset, eval_dataset = create_datasets(
    #     tokenizer,
    #     data_args,
    #     training_args,
    #     apply_chat_template=model_args.chat_template_format != "none",
    # )

    train_dataset = build_qwen2_prompt_dataset(data_args.train_path, tokenizer, data_args.max_seq_length,
                                               data_args.max_src_length, True)

    eval_dataset = build_qwen2_prompt_dataset(data_args.test_path, tokenizer, data_args.max_seq_length,

                                              data_args.max_src_length, True)
    from collections import Counter
    keys = []

    for item in train_dataset:
        keys += list(item.keys())
        # if "labels" not in item:
        #     print(item)
        #     exit()
        # print(f"train_dataset: {item}")
        # break
    print(f"train_dataset: {Counter(keys)}")

    keys = []

    for item in eval_dataset:
        keys += list(item.keys())
        # if "labels" not in item:
        #     print(item)
        #     exit()
        # print(f"train_dataset: {item}")
        # break
    print(f"eval_dataset: {Counter(keys)}")

    for item in eval_dataset:
        # if "labels" not in item:
        #     print(item)
        #     exit()
        print(f"eval_dataset: {item}")
        break
    print(f"=" * 30)
    print(f"training_args:{training_args}")
    print(f"=" * 30)

    # print(f"training_args:{training_args}")

    data_collator = DataCollator(tokenizer)

    # update configs
    dict_args = training_args.to_dict()
    dict_args.pop("push_to_hub_token")
    args = SFTConfig(**dict_args)

    args.packing = data_args.packing
    args.dataset_kwargs = {
        "append_concat_token": data_args.append_concat_token,
        "add_special_tokens": data_args.add_special_tokens,
    }
    args.dataset_text_field = data_args.dataset_text_field,
    args.max_seq_length = data_args.max_seq_length
    args.remove_unused_columns = Fasle
    # data_args.label_names = ["labels"]

    # trainer
    trainer = SFTTrainer(
        model=model,
        # tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        # packing=data_args.packing,
        # dataset_kwargs={
        #     "append_concat_token": data_args.append_concat_token,
        #     "add_special_tokens": data_args.add_special_tokens,
        # },
        # dataset_text_field=data_args.dataset_text_field,
        # max_seq_length=data_args.max_seq_length,
    )

    if model_args.use_peft_lora:
        trainer.accelerator.print(f"{trainer.model}")
        trainer.model.print_trainable_parameters()
        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy
            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # train
    # checkpoint = "/data/work/lcong/ChatGPT/Llama3-Funetuning/peft/qwen1.5-72b-qlora-fsdp/checkpoint-832/"
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    # print("*"*30)
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    print(parser)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
