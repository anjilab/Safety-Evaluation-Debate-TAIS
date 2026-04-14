# Structured Safety Debate

This folder contains a contribution-focused experimental track for evaluator-side safety classification on BeaverTails.

The main change relative to the existing pipeline is that "debate" is made explicit:

- one agent argues that the response is `Unsafe`
- one agent argues that the response is `Safe`
- one judge reads both sides and returns the final verdict

This lets us compare:

- `single`: one evaluator, one judgment
- `vote`: multiple independent evaluators, majority vote
- `role_vote`: multiple role-specialized evaluators, majority vote
- `structured_debate`: explicit opposing-position debate with a final judge

## Why this folder exists

The current main pipeline is useful, but its debate mode is still closer to iterative consensus than true evaluator-side adversarial debate. This folder isolates the main methodological contribution without disturbing the existing code path.

## Example commands

```bash
python structured_safety_debate/run.py --mode single --model qwen2.5-7b --data_size 50
python structured_safety_debate/run.py --mode vote --model qwen2.5-7b --num_agents 5 --data_size 50
python structured_safety_debate/run.py --mode role_vote --model qwen2.5-7b --data_size 50
python structured_safety_debate/run.py --mode structured_debate --model qwen2.5-7b --data_size 50
```

## Output

Each run writes a JSONL file under `out/structured_safety_debate/` and prints:

- overall accuracy
- unsafe precision
- unsafe recall
- unsafe false negative rate
- per-category accuracy
