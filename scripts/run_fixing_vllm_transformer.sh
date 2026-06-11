
# CUDA_VISIBLE_DEVICES=4 .venv/bin/python src/main.py \
#   --data safety_eval \
#   --model llama3.1-8b \
#   --num_agents 5 \
#   --data_size 200 \
#   --debate_rounds 5 \
#   --multi_persona \
#   --inference_backend transformers &


# CUDA_VISIBLE_DEVICES=2 .venv/bin/python src/main_faster.py \
#   --data safety_eval \
#   --model llama3.1-8b \
#   --num_agents 5 \
#   --data_size 200 \
#   --debate_rounds 5 \
#   --multi_persona \
#   --inference_backend vllm \
#   --tensor_parallel_size 1 \
#   --vllm_gpu_memory_utilization 0.9 \
#   --max_model_len 4096

CUDA_VISIBLE_DEVICES=4 python src/main_faster.py \
  --data safety_eval \
  --model llama3.1-8b \
  --num_agents 5 \
  --data_size 200 \
  --debate_rounds 5 \
  --multi_persona \
  --inference_backend transformers \
  --batch_size 1 &


CUDA_VISIBLE_DEVICES=2 python src/main_faster.py \
  --data safety_eval \
  --model llama3.1-8b \
  --num_agents 5 \
  --data_size 200 \
  --debate_rounds 5 \
  --multi_persona \
  --inference_backend vllm \
  --tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.9 \
  --max_model_len 4096 \
  --batch_size 1

