#### vllm 

# CUDA_VISIBLE_DEVICES=0 python src/main_faster.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 0 --solver vote --multi_persona --agent_selection 1d4c --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &
# CUDA_VISIBLE_DEVICES=1 python src/main_faster.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 0 --solver vote --multi_persona --agent_selection 2d3c --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &
# CUDA_VISIBLE_DEVICES=2 python src/main_faster.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 0 --solver vote --multi_persona --agent_selection 3d2c --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &
# CUDA_VISIBLE_DEVICES=3 python src/main_faster.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 0 --solver vote --multi_persona --agent_selection 4d1c --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &

# CUDA_VISIBLE_DEVICES=0 python src/main_faster.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 1d4c --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &
# CUDA_VISIBLE_DEVICES=1 python src/main_faster.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 2d3c --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &
# CUDA_VISIBLE_DEVICES=2 python src/main_faster.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 3d2c --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &
# CUDA_VISIBLE_DEVICES=3 python src/main_faster.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 4d1c --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &

CUDA_VISIBLE_DEVICES=0 python src/main_faster.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 1d4c --sparse --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &
CUDA_VISIBLE_DEVICES=1 python src/main_faster.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 2d3c --sparse --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &
CUDA_VISIBLE_DEVICES=2 python src/main_faster.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 3d2c --sparse --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &
CUDA_VISIBLE_DEVICES=3 python src/main_faster.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 4d1c --sparse --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9 &


wait




#### Transformers
# CUDA_VISIBLE_DEVICES=0 python src/main_old_persona.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 1d4c &
# CUDA_VISIBLE_DEVICES=1 python src/main_old_persona.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 2d3c &
# CUDA_VISIBLE_DEVICES=2 python src/main_old_persona.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 3d2c &
# CUDA_VISIBLE_DEVICES=3 python src/main_old_persona.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 4d1c &


CUDA_VISIBLE_DEVICES=0 python src/main_old_persona.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 1d4c --sparse &
CUDA_VISIBLE_DEVICES=1 python src/main_old_persona.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 2d3c --sparse &
CUDA_VISIBLE_DEVICES=2 python src/main_old_persona.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 3d2c --sparse &
CUDA_VISIBLE_DEVICES=3 python src/main_old_persona.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --debate_rounds 5 --solver vote --multi_persona --agent_selection 4d1c --sparse &


wait
