# Safety Evaluation Using Debate 

In this work, we are exploring safety evaluation using debate.

## Multi-Agent Debate Methods

The key distinction among multi-agent debate methods lies in how agents communicate and the roles they assume. We evaluate the following representative approaches: 

### 1. Decentralized MAD (Default)
**Command:** `python src/main.py --model qwen2.5-7b --num_agents 5 --data safety_eval`

In Decentralized Multi-Agent Debate, each agent observes **all other agents' responses** from the previous round. This creates a fully-connected communication topology where every agent has complete visibility into the entire debate history.

- **Communication Pattern:** All-to-all (fully connected)
- **Agent Behavior:** Each agent considers all peer opinions when revising their answer
- **Use Case:** Maximum information sharing, suitable when you want comprehensive deliberation
- **Parameters:** Set `N` (number of agents) via `--num_agents`, typically N=5

### 2. Sparse MAD
**Command:** `python src/main.py --model qwen2.5-7b --num_agents 5 --data safety_eval --sparse`

Sparse MAD is a variant of Decentralized MAD with a **sparse communication topology** to enhance efficiency. Each agent only observes responses from their immediate neighbors (previous and next agent in the sequence).

- **Communication Pattern:** Ring topology (each agent sees 2 neighbors)
- **Agent Behavior:** Each agent considers only adjacent agents' opinions
- **Use Case:** Reduces token usage and computational cost while maintaining debate dynamics
- **Efficiency:** Significantly fewer tokens per round compared to Decentralized MAD

### 3. Centralized MAD
**Command:** `python src/main.py --model qwen2.5-7b --num_agents 5 --data safety_eval --centralized`

In Centralized MAD, a **central agent** aggregates peer responses and generates the updated response at each round, while other agents only respond to the central agent's opinion.

- **Communication Pattern:** Hub-and-spoke (star topology)
- **Agent Roles:** 
  - Central agent (Agent 1): Sees all peer responses, synthesizes consensus
  - Peripheral agents: Only see the central agent's response
- **Use Case:** Hierarchical decision-making, when one agent should lead the deliberation
- **Final Answer:** Determined by the central agent's verdict

### 4. Majority Voting (Baseline)
**Command:** `python src/main.py --model qwen2.5-7b --num_agents 5 --data safety_eval --debate_rounds 0`

Majority Voting aggregates **initial responses** from multiple agents without any debate. This is equivalent to T=0 rounds of debate—agents provide independent judgments and the final answer is determined by majority vote.

- **Communication Pattern:** None (no interaction between agents)
- **Agent Behavior:** Each agent provides independent initial judgment
- **Use Case:** Baseline comparison, ensemble without deliberation
- **Efficiency:** Most token-efficient approach (single round of inference)

### Heterogeneous Agents (Optional)
**Command:** Add `--multi_persona` to any of the above

When using `--multi_persona`, agents assume different specialized roles (e.g., Critic, Defender, Judge, Analyst, Ethicist for safety evaluation). This creates diversity in perspectives and reasoning approaches.

### Comparison Summary

| Method | Communication | Token Cost | Information Flow | Best For |
|--------|--------------|------------|------------------|----------|
| **Decentralized MAD** | All-to-all | High | Every agent sees all opinions | Maximum deliberation quality |
| **Sparse MAD** | Ring (2 neighbors) | Medium | Limited peer visibility | Balanced efficiency/quality |
| **Centralized MAD** | Hub-and-spoke | Medium | Central aggregation | Hierarchical decision-making |
| **Majority Voting** | None | Low (1 round) | No interaction | Cost-efficient baseline |

For all multi-agent approaches, we use **N=5 agents** by default. You can ablate the effect of N using `--num_agents`. For single-agent baselines, results are averaged across 5 independent runs. This methods are same as debate or vote paper. 

## Requirements Setup

(1) Setup environment

```
uv venv --python 3.10 --seed;
source .venv/bin/activate
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match 
```

(2) Setup Huggingface API

You need to have a huggingface account with a corresponding API credentials saved under a file named ``token``. It should contain a single line of the token string.

(3) Setup Directories

Create an output folder:

```
mkdir out
```

## Run Experiments

The experiment commands for each dataset are provided in ``scripts/``.
For each experiment, you will need to set up a directory to load the LLM parameters and datasets. To do that, add the ``--data_dir`` and ``--model_dir`` arguments to the command with your own directories.

For example, to run the arithmetics dataset on Qwen2.5-7B-Instruct, run

```
CUDA_VISIBLE_DEVICES=0 python src/main.py --model qwen2.5-7b --num_agents 5 --data arithmetics --data_size 100 --debate_rounds 5 --data_dir [your_directory] --model_dir [your_directory]
```

To run Sparse MAD or Centralized MAD, add ``--sparse`` or ``--centralized`` to the command. To run heterogeneous agent settings, add ``--multi_persona``.

By default, experiments now use vLLM for generation-only inference. You can tune vLLM with:

```
python src/main.py --model qwen2.5-7b --data safety_eval --data_size 100 --tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.9
```

To fall back to the previous Hugging Face Transformers inference path, pass ``--inference_backend transformers``.

## Acknowledgement

We thank debate or vote paper for their code from which we explored our work.

```
@inproceedings{choi2025debate,
  title={Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?},
  author={Choi, Hyeong Kyu and Zhu, Xiaojin and Li, Sharon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
