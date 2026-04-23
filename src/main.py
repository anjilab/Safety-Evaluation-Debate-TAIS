
import argparse, sys, os, copy, time, random, json, pickle, re, collections, gc
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import torch
from rouge_score import rouge_scorer
import wandb
ROUGE = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])


from model.model_utils import get_agents, engine
from data.data_utils import load_data
from evaluator import get_instruction_suffix, evaluate_arithmetics, evaluate_mcq, base_evaluate_arithmetics, base_evaluate_mcq, evaluate_gen, evaluate_safety



class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serializable")


def get_args():

    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default="out/")
    parser.add_argument('--wandb_project', type=str, default="safety-evaluation-debate")
    parser.add_argument('--wandb_entity', type=str, default=None)

    # data
    parser.add_argument('--data_dir', type=str, default="/media/drive1/anjila/codes/Safety-Evaluation-Debate-TAIS/data_dir")
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--sub_data', type=str, default='')
    parser.add_argument('--data_size', type=int, default=0)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--debug', action='store_true')
    # agent
    parser.add_argument('--num_agents', type=int, default=5)

    parser.add_argument('--agent_selection', type=str, default="none")
    parser.add_argument('--multi_persona', action='store_true')


    # model
    parser.add_argument('--model', type=str, default="llama3.1-8b")
    parser.add_argument('--model_dir', type=str, default="/media/drive1/anjila/codes/Safety-Evaluation-Debate-TAIS/model_dir")
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--inference_backend', type=str, default="vllm", choices=["vllm", "transformers"])
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_model_len', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')


    # debate
    parser.add_argument('--debate_rounds', type=int, default=5)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--centralized', action='store_true')

    parser.add_argument('--solver', type=str, default='vote', choices=['vote','debate'])
    parser.add_argument('--generate_first_round', action='store_true')
    parser.add_argument('--max_num_agents', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--bae', action='store_true', help="base answer extractor")
    parser.add_argument('--cot', action='store_true')


    return parser.parse_args()


def get_new_message(args, sample, responses, personas=None, suffix=None):

    new_message = {}
    agents = list(responses.keys())
    if len(agents) > 1 : # MULTI-AGENT DEBATE

        if not args.centralized : # DECENTRALIZED MAD
            for i, agent in enumerate(agents) :
                msg = "These are the recent opinions from other agents: "
                if args.sparse :
                    peers = [agents[(i-1) % len(agents)], agents[(i+1) % len(agents)]]
                else :
                    peers = agents[:i]+agents[i+1:]
                for other_agent in peers:
                    msg += f"\n\nOne of the agents' response: \n{responses[other_agent]}\n"
                msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
                msg += f'\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}'

                if suffix is not None :
                    msg += suffix

                if personas is not None :
                    new_message[agent] = [{'role': 'system', 'content': personas[agent.split("__")[-2]]},{'role': 'user', 'content': msg}]
                else :
                    new_message[agent] = {'role': 'user', 'content': msg}

        else : # CENTRALIZED MAD
            for i, agent in enumerate(agents):
                if i == 0 :
                    msg = "These are the recent opinions from other agents: "
                    peers = agents[:i]+agents[i+1:]
                    for other_agent in peers:
                        msg += f"\n\nOne of the agents' response: \n{responses[other_agent]}\n"
                    msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
                    msg += f'\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}'
                else :
                    msg = f"This is the recent opinion from another agent: \n{responses[agents[0]]}\n"
                    msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
                    msg += f'\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}'
                
                if suffix is not None :
                    msg += suffix

                if personas is not None :
                    new_message[agent] = [{'role': 'system', 'content': personas[agent.split("__")[-2]]},{'role': 'user', 'content': msg}]
                else :
                    new_message[agent] = {'role': 'user', 'content': msg}

    else : # SINGLE AGENT SELF REFINEMENT
        for i, agent in enumerate(agents) :
            msg = f"This was your most recent opinion:\n{responses[agents[i]]}\n"
            msg += f'\n\nRevise your recent opinion to give your updated final answer to the question:\n{sample}'

            if suffix is not None :
                msg += suffix

            if personas is not None :
                new_message[agent] = [{'role': 'system', 'content': personas[agent.split("__")[-2]]},{'role': 'user', 'content': msg}]
            else :
                new_message[agent] = {'role': 'user', 'content': msg}

    return new_message


def main(args):

    '''
    Steps for ANJILA ONLY:
    1. Load Agents with and without personas depending on args.multi_persona
    2. Load Data: Our case is safety evaluation data only. 
    3. Then folder and file saving setup
    4. Evaluation function setup depending on args.data and args.bae
    5. Debate loop: 
        1. For each question, gather initial opinions from agents. 
        2. Based on persona, message append is different.

    '''

    # Initialize Weights & Biases
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.data}_{args.data_size}__{args.model}_N={args.num_agents}_R={args.debate_rounds}",
        config=vars(args)
    )

    # Load Agents
    agent, personas = get_agents(args) # AGENT = MODEL Object, personas = this is only used when multi_persona is true. 

    # Load Data
    test_X, test_Y = load_data(args, split='test')


    # Setup Names
    fname = f"{args.data}_{args.data_size}__{args.model}_N={args.num_agents}_R={args.debate_rounds}"
    if args.sparse : fname += '_SPARSE'
    elif args.centralized : fname += '_CENTRAL'
    if args.bae : fname += '_BAE'
    if args.multi_persona : fname += '_HETERO'

    agent_names = []
    for i in range(args.num_agents):
        for persona in personas.keys():
            agent_names.append(f"{args.data}_{args.data_size}__{args.model}__{persona}__Agent{i+1}")
          

    # Setup Experiments
    SUFFIX = get_instruction_suffix(args)

    if args.data in ['arithmetics','gsm8k']:
        if args.bae :
            evaluate = base_evaluate_arithmetics
        else :
            evaluate = evaluate_arithmetics
    elif args.data in ['hellaswag','pro_medicine','formal_logic','csqa','hh_rlhf']:
        if args.bae:
            evaluate = base_evaluate_mcq
        else :
            evaluate = evaluate_mcq
    elif args.data in ['safety_eval']:
        if args.bae:
            evaluate = base_evaluate_mcq
        else:
            evaluate = evaluate_safety
    elif args.data in ['cnn_daily'] :
        evaluate = evaluate_gen
    else :
        raise NotImplementedError

    
    # Debate
    sample_responses = []
    iscorr_list = []
    total_input_tokens = 0
    total_output_tokens = 0
    round_token_stats = []  # Track tokens per round for cost analysis


    for i, (x, y) in tqdm(enumerate(zip(test_X, test_Y)), total=len(test_X)):

        print('\n\nQuestion: ', x + SUFFIX, '\n')

        # initialize opinions
        print("Gathering initial opinions...")
        round_iscorr = []
        if args.multi_persona :
            messages = []
            for name, sys in personas.items():
                messages.append([{"role": "system", "content": sys},{"role": "user", "content": x + SUFFIX}])
        else:
            messages = [{"role": "user", "content": x + SUFFIX}] * args.num_agents
        responses, input_tokens, output_tokens = engine(messages, agent, args.num_agents)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        agent_responses = dict(zip(agent_names, responses))
        print(messages, responses, agent_responses)

        # evaluate
        if args.centralized :
            central_agent_response = {list(agent_responses.keys())[0] : list(agent_responses.values())[0]}
            final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)
        else :
            final_resps, debate_resps, is_corr = evaluate(agent_responses, y)

        print(f"ROUND 0 : {final_resps} (answer = {y})")
        if args.data in ['arithmetics','gsm8k']:
            round_data = {
                'responses': agent_responses,
                'final_answers': final_resps,
                'final_answer_iscorr': [y_pred == np.round(y,1) for y_pred in final_resps],
                'debate_answer': debate_resps,
                'debate_answer_iscorr': is_corr,
                'answer': np.round(y, 1),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
            }
        elif args.data in ['safety_eval']:
            round_data = {
                'responses': agent_responses,
                'final_answers': final_resps,
                'final_answer_iscorr': [y_pred == y for y_pred in final_resps],
                'debate_answer': debate_resps,
                'debate_answer_iscorr': is_corr,
                'answer': y,
                'category': args.safety_categories[i],
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
            }
        else :
            round_data = {
                'responses': agent_responses,
                'final_answers': final_resps,
                'final_answer_iscorr': [y_pred == y for y_pred in final_resps],
                'debate_answer': debate_resps,
                'debate_answer_iscorr': is_corr,
                'answer': y,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
            }
        rounds_data_dict = {'0': round_data}
        round_iscorr.append(is_corr)
        
        # Log Round 0 to wandb with individual agent verdicts
        wandb_log = {
            f'sample_{i}/round_0/accuracy': float(is_corr),
            f'sample_{i}/round_0/debate_answer': debate_resps,
            f'sample_{i}/round_0/input_tokens': input_tokens,
            f'sample_{i}/round_0/output_tokens': output_tokens,
        }
        # Log each agent's verdict
        for idx, (agent_name, answer) in enumerate(zip(agent_names, final_resps)):
            agent_correct = round_data['final_answer_iscorr'][idx]
            wandb_log[f'sample_{i}/round_0/agent_verdict/{agent_name}'] = str(answer)
            wandb_log[f'sample_{i}/round_0/agent_correct/{agent_name}'] = float(agent_correct)
        for agent_name, response in agent_responses.items():
            wandb_log[f'sample_{i}/round_0/response/{agent_name}'] = response
        if args.data in ['safety_eval']:
            wandb_log[f'sample_{i}/category'] = args.safety_categories[i]
        wandb.log(wandb_log)
        
        # Track previous round answers for flip detection
        prev_answers = final_resps.copy()

        start = 1


        # begin debate
        for r in range(start, args.debate_rounds+1) :

            print(f"Debating round {r}...")
            if args.multi_persona:
                new_agent_messages = get_new_message(args, x, agent_responses, personas, suffix=SUFFIX)
            else:
                new_agent_messages = get_new_message(args, x, agent_responses, suffix=SUFFIX)
            messages = list(new_agent_messages.values())
            responses, input_tokens, output_tokens = engine(messages, agent, args.num_agents)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            agent_responses = dict(zip(agent_names, responses))

            # evaluate
            if args.centralized:
                central_agent_response = {list(agent_responses.keys())[0] : list(agent_responses.values())[0]}
                final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)
            else :
                final_resps, debate_resps, is_corr = evaluate(agent_responses, y)

            print("\n\n" + str(messages[0]) + "\n\n")
            print(f"ROUND {r} : {final_resps} (answer = {y})")
            if args.data in ['arithmetics','gsm8k']:
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == np.round(y,1) for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': np.round(y, 1),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                }
            elif args.data in ['safety_eval']:
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == y for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': y,
                    'category': args.safety_categories[i],
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                }
            elif args.data in ['cnn_daily'] :
                scores = []
                for summary in final_resps:
                    s = ROUGE.score(y, summary)
                    rouge1 = s['rouge1'].fmeasure
                    rouge2 = s['rouge2'].fmeasure
                    rougeL = s['rougeL'].fmeasure
                    scores.append((rouge1, rouge2, rougeL))
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': scores,
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': y,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                }
            else :
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == y for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': y,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                }
            rounds_data_dict[str(r)] = round_data
            round_iscorr.append(is_corr)
            
            # Log this round to wandb with agent verdicts and flip detection
            wandb_log = {
                f'sample_{i}/round_{r}/accuracy': float(is_corr),
                f'sample_{i}/round_{r}/debate_answer': debate_resps,
                f'sample_{i}/round_{r}/input_tokens': input_tokens,
                f'sample_{i}/round_{r}/output_tokens': output_tokens,
            }
            # Log each agent's verdict and detect flips
            for idx, (agent_name, answer) in enumerate(zip(agent_names, final_resps)):
                agent_correct = round_data['final_answer_iscorr'][idx]
                wandb_log[f'sample_{i}/round_{r}/agent_verdict/{agent_name}'] = str(answer)
                wandb_log[f'sample_{i}/round_{r}/agent_correct/{agent_name}'] = float(agent_correct)
                # Detect if agent flipped answer
                if prev_answers[idx] != answer:
                    wandb_log[f'sample_{i}/round_{r}/agent_flipped/{agent_name}'] = 1
                    print(f"  └─ {agent_name} FLIPPED: {prev_answers[idx]} → {answer}")
                else:
                    wandb_log[f'sample_{i}/round_{r}/agent_flipped/{agent_name}'] = 0
            for agent_name, response in agent_responses.items():
                wandb_log[f'sample_{i}/round_{r}/response/{agent_name}'] = response
            wandb.log(wandb_log)
            
            # Update previous answers for next round
            prev_answers = final_resps.copy()

        sample_responses.append(rounds_data_dict)
        iscorr_list.append(round_iscorr)

        # Save to jsonl
        print(len(sample_responses))
        with open(f'out/history/{fname}.jsonl', 'w') as f:
            for record in sample_responses:
                f.write(json.dumps(record, default=convert_numpy) + '\n')
            
        if args.data in ['cnn_daily'] :
            rouge1s, rouge2s, rougeLs = [], [], []
            for i in range(len(iscorr_list[0])):
                for _, rouges in enumerate(iscorr_list):
                    rouge1s.append(rouges[i][0])
                    rouge2s.append(rouges[i][1])
                    rougeLs.append(rouges[i][2])
                r1, r2, rL = np.mean(rouge1s), np.mean(rouge2s), np.mean(rougeLs)
                print(f'Round {i} R1: {r1:.4f} / R2: {r2:.4f} / RL: {rL:.4f}')
            round_accs = (r1, r2, rL)
        elif args.data in ['safety_eval']:
            round_accs = np.array(iscorr_list).mean(0)
            for idx, acc in enumerate(round_accs):
                print(f'Round {idx} Acc.: {acc:.4f}')
                wandb.log({f'overall/round_{idx}/accuracy': float(acc)})
            # Per-category accuracy (final round only)
            cat_correct = collections.defaultdict(list)
            for sample_rounds, cat in zip(iscorr_list, args.safety_categories):
                cat_correct[cat].append(sample_rounds[-1])
            print('\nPer-category accuracy (final round):')
            for cat, corrects in sorted(cat_correct.items()):
                print(f'  {cat}: {np.mean(corrects):.4f} (n={len(corrects)})')
                wandb.log({f'per_category/{cat}/final_accuracy': float(np.mean(corrects)),
                          f'per_category/{cat}/sample_count': len(corrects)})
        else :
            round_accs = np.array(iscorr_list).mean(0)
            for i, acc in enumerate(round_accs) :
                print(f'Round {i} Acc.: {acc:.4f}')
                wandb.log({f'overall/round_{i}/accuracy': float(acc)})
    
    # Log final summary with token costs
    total_tokens = total_input_tokens + total_output_tokens
    avg_tokens_per_sample = total_tokens / len(test_X) if len(test_X) > 0 else 0
    
    print(f'\n=== Token Usage Summary ===')
    print(f'Total Input Tokens: {total_input_tokens:,}')
    print(f'Total Output Tokens: {total_output_tokens:,}')
    print(f'Total Tokens: {total_tokens:,}')
    print(f'Avg Tokens per Sample: {avg_tokens_per_sample:.1f}')
    print(f'Rounds per Sample: {args.debate_rounds + 1}')
    print(f'Avg Tokens per Round: {avg_tokens_per_sample / (args.debate_rounds + 1):.1f}')
    
    wandb.log({
        'final_accuracy': float(round_accs[-1]),
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_tokens,
        'avg_tokens_per_sample': avg_tokens_per_sample,
        'avg_tokens_per_round': avg_tokens_per_sample / (args.debate_rounds + 1),
    })
    wandb.finish()
    
    with open('out/logs.tsv', 'a') as f :
        line = f"\n{args.timestamp}\t{fname}\t{round_accs}"
        f.writelines(line)





if __name__ == "__main__":
    
    args = get_args()
    log_prefix = f"{args.data or 'run'}_{args.model}_seed{args.seed}"
    os.makedirs(os.path.join("out", "logs"), exist_ok=True)
    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("out", "logs", f"{log_time}_{log_prefix}.log")
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)
        print(f"Terminal log: {log_path}")
        print("Command: " + " ".join(sys.argv))

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if args.inference_backend != "vllm": # Without this issuse of spawning process of vllm persists. 
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        args.timestamp = timestamp

        with open('token','r') as f :
            token = f.read()
        args.token = token

        main(args)
    
