torchrun --nnodes 1 --nproc_per_node 2 ../training_reward_model.py \
    --model_name '/iyunwen/data/public_pretrain_models/Qwen2.5-0.5B-Instruct/' \
    --dataset_name '../data/comparison_data.json' \
    --output_dir '../checkpoints/training_reward_model/'