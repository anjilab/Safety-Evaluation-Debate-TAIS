# vote vs debate comparison (the main gap vs the paper)
# python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --solver vote
# python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --solver debate --debate_rounds 3

# # with safety-specific roles (critic/defender/judge structure)
# python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --solver debate --debate_rounds 3 --multi_persona

# # topology experiments
# python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --solver debate --sparse
# python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --solver debate --centralized

# python src/main.py --data safety_eval --model llama3.1-8b --num_agents 5 --data_size 200
# python src/main.py --data safety_eval --model llama3.1-8b --num_agents 5 --data_size 200 --solver debate --centralized
# python src/main.py --data safety_eval --model llama3.1-8b --num_agents 5 --data_size 200 --solver debate --sparse
# python src/main.py --data safety_eval --model llama3.1-8b --num_agents 5 --data_size 200 --solver debate --multi_persona

## For VLLM
# python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --solver debate --sparse --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9
# python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 200 --solver debate --debate_rounds 5 --multi_persona --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9


### Using VLLM and experiments to run: 

CUDA_VISIBLE_DEVICES=0 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 0 --solver vote --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9
CUDA_VISIBLE_DEVICES=1 python src/main.py --data safety_eval --model qwen2.5-7b --num_agents 5 --data_size 500 --debate_rounds 0 --solver vote --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9
