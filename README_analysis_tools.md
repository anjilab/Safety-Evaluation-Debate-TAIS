# Analysis Tools for Safety Evaluation Debate

This document describes the three analysis tools added to track and analyze multi-agent debate performance.

## 1. Round-by-Round Agent Verdict Logger

### What it does
Tracks every agent's verdict at every round and detects when agents "flip" their answers (change from correct to incorrect or vice versa).

### Where to find it
Built into `src/main.py` - automatically logs to both:
- **JSONL files**: `out/history/{experiment_name}.jsonl`
- **Weights & Biases**: Real-time tracking with visualizations

### Key metrics tracked
- Individual agent verdicts per round
- Agent correctness per round
- Agent answer flips between rounds
- Input/output token counts per round

### Example output
```
ROUND 1 : ['(A)', '(B)', '(A)', '(B)', '(A)'] (answer = (B))
  └─ safety_eval_200__qwen2.5-7b__Critic__Agent1 FLIPPED: (A) → (B)
  └─ safety_eval_200__qwen2.5-7b__Judge__Agent3 FLIPPED: (A) → (B)
```

### Accessing in wandb
Each run logs:
- `sample_{i}/round_{r}/agent_verdict/{agent_name}` - The agent's answer
- `sample_{i}/round_{r}/agent_correct/{agent_name}` - Whether it was correct (0 or 1)
- `sample_{i}/round_{r}/agent_flipped/{agent_name}` - Whether agent changed answer (0 or 1)

### Data structure in JSONL
```json
{
  "0": {
    "responses": {...},
    "final_answers": ["(A)", "(B)", "(A)", "(B)", "(A)"],
    "final_answer_iscorr": [false, true, false, true, false],
    "debate_answer": "(A)",
    "debate_answer_iscorr": false,
    "answer": "(B)",
    "category": "hate_speech,offensive_language",
    "input_tokens": 1234,
    "output_tokens": 567
  },
  "1": { ... },
  ...
}
```

## 2. Category Aggregator Script

### What it does
Analyzes per-category accuracy across all rounds from saved debate results. Shows which harm categories improve or degrade through debate.

### Usage
```bash
python analyze_categories.py out/history/safety_eval_200__qwen2.5-7b_N=5_R=5.jsonl
```

### Output example
```
======================================================================
Category Analysis: safety_eval_200__qwen2.5-7b_N=5_R=5.jsonl
Total Samples: 200
======================================================================

Round 0:
  Category                                      Accuracy     Count    Flips   
  ---------------------------------------------------------------------------
  animal_abuse                                   82.4%       17       
  child_abuse                                    91.7%       12       
  controversial_topics,politics                  76.5%       17       
  discrimination,stereotype,injustice            68.8%       16       
  drug_abuse,weapons,banned_substance            85.7%       14       
  ...
  ---------------------------------------------------------------------------
  OVERALL                                        79.5%       200      

Round 1:
  Category                                      Accuracy     Count    Flips   
  ---------------------------------------------------------------------------
  animal_abuse                                   88.2%       17       3       
  child_abuse                                    91.7%       12       0       
  ...

======================================================================
Category Improvement (First vs Last Round)
======================================================================
  Category                                      First      Last       Change    Count   
  -------------------------------------------------------------------------------------
  animal_abuse                                   82.4%      88.2%      +5.8%    17      
  self_harm                                      75.0%      81.2%      +6.2%    16      
  ...
```

### Key insights
- **Per-category accuracy**: See which harm categories the model struggles with
- **Category improvement**: Identify which categories benefit most from debate
- **Flip tracking**: Understand debate dynamics and consensus formation

## 3. Token Cost Calculator

### What it does
Compares token usage and costs across different debate configurations (e.g., 1-round vs 5-round debates) to calculate ROI.

### Usage
```bash
# Compare single configuration
python calculate_token_costs.py out/history/safety_eval_200__qwen2.5-7b_N=5_R=5.jsonl

# Compare multiple configurations
python calculate_token_costs.py \
    out/history/safety_eval_200__qwen2.5-7b_N=5_R=1.jsonl \
    out/history/safety_eval_200__qwen2.5-7b_N=5_R=5.jsonl
```

### Output example
```
====================================================================================================
TOKEN USAGE COMPARISON
====================================================================================================

Configuration                                      Rounds   Samples    Total Tokens    Tokens/Sample  
----------------------------------------------------------------------------------------------------
safety_eval_200__qwen2.5-7b_N=5_R=1.jsonl         2        200        234,567         1,173          
safety_eval_200__qwen2.5-7b_N=5_R=5.jsonl         6        200        1,523,890       7,619          

====================================================================================================
ROI ANALYSIS: Multi-Round Debate vs Baseline
====================================================================================================

Baseline Configuration: safety_eval_200__qwen2.5-7b_N=5_R=1.jsonl
  - Rounds: 2
  - Total Tokens: 234,567
  - Tokens/Sample: 1,173
  - Estimated Cost: $2.81
  - Cost/Sample: $0.0141

Multi-Round Configuration: safety_eval_200__qwen2.5-7b_N=5_R=5.jsonl
  - Rounds: 6
  - Total Tokens: 1,523,890
  - Tokens/Sample: 7,619
  - Estimated Cost: $18.29
  - Cost/Sample: $0.0914

Impact:
  - Token increase: 6.50x
  - Cost increase: 6.51x
  - Additional cost per sample: $0.0773

====================================================================================================
DETAILED TOKEN BREAKDOWN
====================================================================================================

Configuration                                      Input Tokens    Output Tokens   I/O Ratio 
----------------------------------------------------------------------------------------------------
safety_eval_200__qwen2.5-7b_N=5_R=1.jsonl         156,789         77,778          2.02      
safety_eval_200__qwen2.5-7b_N=5_R=5.jsonl         1,015,927       507,963         2.00      

Note: Cost estimates use approximate API pricing ($0.003/1K input, $0.015/1K output).
Adjust rates in the script for your specific model pricing.
```

### Use cases
- **Workshop ROI statement**: Show concrete token/cost tradeoffs for different debate rounds
- **Configuration optimization**: Find the sweet spot between accuracy and cost
- **Budget planning**: Estimate costs before running large-scale experiments

## Weights & Biases Integration

All metrics are automatically logged to wandb during training:

### Configure wandb
```bash
python src/main.py \
    --data safety_eval \
    --data_size 200 \
    --debate_rounds 5 \
    --num_agents 5 \
    --wandb_project "safety-evaluation-debate" \
    --wandb_entity "your-username"
```

### Key wandb metrics
- `overall/round_{r}/accuracy` - Aggregate accuracy per round
- `per_category/{category}/final_accuracy` - Per-category final accuracy
- `sample_{i}/round_{r}/agent_verdict/{agent}` - Individual agent decisions
- `sample_{i}/round_{r}/agent_flipped/{agent}` - Track answer changes
- `total_tokens`, `total_input_tokens`, `total_output_tokens` - Cost tracking
- `avg_tokens_per_sample`, `avg_tokens_per_round` - Token efficiency

## Quick Start Example

1. **Run baseline (1 round)**:
```bash
python src/main.py --data safety_eval --data_size 200 --debate_rounds 0 --num_agents 5 --seed 42
```

2. **Run multi-round debate (5 rounds)**:
```bash
python src/main.py --data safety_eval --data_size 200 --debate_rounds 5 --num_agents 5 --seed 42
```

3. **Analyze per-category performance**:
```bash
python analyze_categories.py out/history/safety_eval_200__qwen2.5-7b_N=5_R=5.jsonl
```

4. **Calculate token costs & ROI**:
```bash
python calculate_token_costs.py \
    out/history/safety_eval_200__qwen2.5-7b_N=5_R=0.jsonl \
    out/history/safety_eval_200__qwen2.5-7b_N=5_R=5.jsonl
```

## Files Added/Modified

### Modified
- `src/main.py` - Added token tracking, agent verdict logging, flip detection
- `src/model/model_utils.py` - Added token counting to engine functions
- `requirements.txt` - Added `wandb` dependency

### New scripts
- `analyze_categories.py` - Per-category accuracy analysis
- `calculate_token_costs.py` - Token usage and cost comparison
- `README_analysis_tools.md` - This documentation

## Dependencies

Install additional requirements:
```bash
pip install wandb
```

Or use the updated requirements.txt:
```bash
pip install -r requirements.txt
```
