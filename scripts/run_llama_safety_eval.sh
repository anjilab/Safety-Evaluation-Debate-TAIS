#!/bin/bash
# ============================================================
# GPU 1 — Llama-3.1-8B (all conditions, runs in parallel with GPU 0)
# ============================================================

# 1. Single agent
CUDA_VISIBLE_DEVICES=1 python src/main.py \
  --data safety_eval --model llama-3.1-8b \
  --num_agents 1 --data_size 500 --debate_rounds 0 \
  --solver vote \
  --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &

wait

# 2. Vote (5 agents, no communication)
CUDA_VISIBLE_DEVICES=1 python src/main.py \
  --data safety_eval --model llama-3.1-8b \
  --num_agents 5 --data_size 500 --debate_rounds 0 \
  --solver vote \
  --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &

wait

# 3. HETERO Debate
CUDA_VISIBLE_DEVICES=1 python src/main.py \
  --data safety_eval --model llama-3.1-8b \
  --num_agents 5 --data_size 500 --debate_rounds 5 \
  --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &

wait

# 4. SPARSE Debate
CUDA_VISIBLE_DEVICES=1 python src/main.py \
  --data safety_eval --model llama-3.1-8b \
  --num_agents 5 --data_size 500 --debate_rounds 5 \
  --sparse \
  --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &

wait

# 5. CENTRAL Debate
CUDA_VISIBLE_DEVICES=1 python src/main.py \
  --data safety_eval --model llama-3.1-8b \
  --num_agents 5 --data_size 500 --debate_rounds 5 \
  --centralized \
  --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &

wait

# 6. Role-Vote
CUDA_VISIBLE_DEVICES=1 python src/main.py \
  --data safety_eval --model llama-3.1-8b \
  --num_agents 5 --data_size 500 --debate_rounds 0 \
  --solver vote --multi_persona \
  --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &

wait

# 7. Role-Debate
CUDA_VISIBLE_DEVICES=1 python src/main.py \
  --data safety_eval --model llama-3.1-8b \
  --num_agents 5 --data_size 500 --debate_rounds 5 \
  --multi_persona \
  --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &

wait

echo "Llama done"