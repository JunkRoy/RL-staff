#!/bin/bash

# Define variables
CUDA_DEVICES="0,1,2,3"
PYTHON_EXEC="python3"
MODULE="vllm.entrypoints.openai.api_server"
HOST="0.0.0.0"
PORT=18490
MODEL_PATH=""
LOG_FILE="/home/service/logs/llm_log.log"
TRUST_REMOTE_CODE="--trust-remote-code"
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.95
DTYPE="float16"
SERVED_MODEL_NAME="llm"

# Check if the model path exists
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model path $MODEL_PATH does not exist."
  exit 1
fi

# Check if the log directory exists, if not create it
LOG_DIR=$(dirname "$LOG_FILE")
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to create log directory $LOG_DIR."
    exit 1
  fi
fi

# Start the LLM service
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup $PYTHON_EXEC -m $MODULE \
  --host $HOST \
  --port $PORT \
  --model $MODEL_PATH \
  $TRUST_REMOTE_CODE \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --dtype=$DTYPE \
  --served-model-name $SERVED_MODEL_NAME > $LOG_FILE 2>&1 &

# Check if the command was successful
if [ $? -eq 0 ]; then
  echo "LLM Service started successfully. Logs can be found in $LOG_FILE."
else
  echo "Error: Failed to start the LLM Service."
  exit 1
fi
