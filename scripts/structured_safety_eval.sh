# Single evaluator baseline
python structured_safety_debate/run.py --mode single --model qwen2.5-7b --data_size 200

# Independent majority-vote baseline
python structured_safety_debate/run.py --mode vote --model qwen2.5-7b --num_agents 5 --data_size 200

# Role-based vote baseline
python structured_safety_debate/run.py --mode role_vote --model qwen2.5-7b --data_size 200

# Contribution condition: explicit unsafe-vs-safe debate with a final judge
python structured_safety_debate/run.py --mode structured_debate --model qwen2.5-7b --debate_rounds 1 --data_size 200
