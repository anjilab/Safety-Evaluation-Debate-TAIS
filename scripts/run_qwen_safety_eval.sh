#!/bin/bash
set -e  # exit script immediately if any command fails
# ============================================================
# GPU 0 — Qwen2.5-7B (all conditions)
# ============================================================



# 1. Single agent
# CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 1 --data_size 500 --debate_rounds 0 --solver vote 

# # 2. Vote (5 agents, no communication)
# CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 0 --solver vote 

# # 3. HETERO Debate (5 agents, 5 rounds, fully connected)
# CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 5 

# # 4. SPARSE Debate
# CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 5 --sparse 

# # 5. CENTRAL Debate
# CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 5 --centralized 

# # 6. Role-Vote (multi_persona, no communication)
# CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 0 --solver vote --multi_persona 

# # 7. Role-Debate (multi_persona + debate)
# CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 5 --multi_persona 


##### VLLM SETUP #####

# 1. Single agent
CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 1 --data_size 500 --debate_rounds 0 --solver vote --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 

# 2. Vote (5 agents, no communication)
CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 0 --solver vote --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 

# 3. HETERO Debate (5 agents, 5 rounds, fully connected)
CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 5 --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 

# 4. SPARSE Debate
CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 5 --sparse --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 

# 5. CENTRAL Debate
CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 5 --centralized --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 

# 6. Role-Vote (multi_persona, no communication)
CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 0 --solver vote --multi_persona --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 

# 7. Role-Debate (multi_persona + debate)
CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 5 --multi_persona --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &

wait

echo "Qwen done"