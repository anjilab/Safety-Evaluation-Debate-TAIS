# Safety Evaluation Using Debate 

In this work, we are exploring safety evaluation using debate..

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
