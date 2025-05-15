accelerate launch --multi_gpu --num_machines 1  --num_processes 8 \
    tuning_lm_with_rl.py \
    --log_with wandb \
    --model_name <LLAMA_FINETUNED_MODEL> \
    --reward_model_name <LLAMA_RM_MODEL> \
    --adafactor False \
    --tokenizer_name <LLAMA_TOKENIZER> \
    --save_freq 100 \
    --output_max_length 128 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --batched_gen True \
    --ppo_epochs 4 \
    --learning_rate 1.4e-5 \
    --early_stopping True \
    --output_dir './checkpoints/tuning_llama_rl/'