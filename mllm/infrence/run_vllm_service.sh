CUDA_VISIBLE_DEVICES=1 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /data/public_pretrain_models/Qwen2.5-VL-7B-Instruct \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --port 18400 \
  --served-model-name mllm \
  --dtype bfloat16 > ./log.log 2>&1 &