#!/usr/bin/env python3
"""
Token Cost Calculator: Compare token usage across different debate configurations.
Usage: python calculate_token_costs.py <results_file.jsonl> [results_file2.jsonl ...]
"""
import json
import sys
import os


def analyze_token_costs(jsonl_file):
    """Extract token usage statistics from results file."""
    
    # Load results
    results = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    if not results:
        return None
    
    # Extract configuration from filename
    basename = os.path.basename(jsonl_file)
    parts = basename.replace('.jsonl', '').split('_')
    
    # Count rounds and tokens
    num_rounds = len(results[0])
    total_input = 0
    total_output = 0
    
    for sample in results:
        for round_key in sample:
            round_data = sample[round_key]
            total_input += round_data.get('input_tokens', 0)
            total_output += round_data.get('output_tokens', 0)
    
    total_tokens = total_input + total_output
    num_samples = len(results)
    
    return {
        'file': basename,
        'num_rounds': num_rounds,
        'num_samples': num_samples,
        'total_input_tokens': total_input,
        'total_output_tokens': total_output,
        'total_tokens': total_tokens,
        'avg_tokens_per_sample': total_tokens / num_samples,
        'avg_tokens_per_round': total_tokens / (num_samples * num_rounds),
        'input_output_ratio': total_input / total_output if total_output > 0 else 0,
    }


def format_number(num):
    """Format number with commas."""
    return f"{int(num):,}"


def calculate_costs(stats, cost_per_1k_input=0.003, cost_per_1k_output=0.015):
    """Calculate estimated costs based on token usage.
    Default rates are approximate for many LLM APIs (adjust as needed).
    """
    input_cost = (stats['total_input_tokens'] / 1000) * cost_per_1k_input
    output_cost = (stats['total_output_tokens'] / 1000) * cost_per_1k_output
    total_cost = input_cost + output_cost
    
    return {
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'cost_per_sample': total_cost / stats['num_samples'],
    }


def print_comparison(all_stats):
    """Print comparison table for multiple configurations."""
    
    print(f"\n{'='*100}")
    print("TOKEN USAGE COMPARISON")
    print(f"{'='*100}\n")
    
    # Header
    print(f"{'Configuration':<50} {'Rounds':<8} {'Samples':<10} {'Total Tokens':<15} {'Tokens/Sample':<15}")
    print(f"{'-'*100}")
    
    for stats in all_stats:
        print(f"{stats['file']:<50} {stats['num_rounds']:<8} {stats['num_samples']:<10} "
              f"{format_number(stats['total_tokens']):<15} {format_number(stats['avg_tokens_per_sample']):<15}")
    
    print()
    
    # ROI Analysis
    if len(all_stats) >= 2:
        print(f"\n{'='*100}")
        print("ROI ANALYSIS: Multi-Round Debate vs Baseline")
        print(f"{'='*100}\n")
        
        # Find baseline (1 round) and best multi-round
        baseline = min(all_stats, key=lambda x: x['num_rounds'])
        multi_round = max(all_stats, key=lambda x: x['num_rounds'])
        
        token_increase = multi_round['total_tokens'] / baseline['total_tokens']
        token_per_sample_increase = multi_round['avg_tokens_per_sample'] / baseline['avg_tokens_per_sample']
        
        # Calculate costs
        baseline_costs = calculate_costs(baseline)
        multi_costs = calculate_costs(multi_round)
        cost_increase = multi_costs['total_cost'] / baseline_costs['total_cost']
        
        print(f"Baseline Configuration: {baseline['file']}")
        print(f"  - Rounds: {baseline['num_rounds']}")
        print(f"  - Total Tokens: {format_number(baseline['total_tokens'])}")
        print(f"  - Tokens/Sample: {format_number(baseline['avg_tokens_per_sample'])}")
        print(f"  - Estimated Cost: ${baseline_costs['total_cost']:.2f}")
        print(f"  - Cost/Sample: ${baseline_costs['cost_per_sample']:.4f}\n")
        
        print(f"Multi-Round Configuration: {multi_round['file']}")
        print(f"  - Rounds: {multi_round['num_rounds']}")
        print(f"  - Total Tokens: {format_number(multi_round['total_tokens'])}")
        print(f"  - Tokens/Sample: {format_number(multi_round['avg_tokens_per_sample'])}")
        print(f"  - Estimated Cost: ${multi_costs['total_cost']:.2f}")
        print(f"  - Cost/Sample: ${multi_costs['cost_per_sample']:.4f}\n")
        
        print(f"Impact:")
        print(f"  - Token increase: {token_increase:.2f}x")
        print(f"  - Cost increase: {cost_increase:.2f}x")
        print(f"  - Additional cost per sample: ${multi_costs['cost_per_sample'] - baseline_costs['cost_per_sample']:.4f}")
        
    # Detailed breakdown
    print(f"\n{'='*100}")
    print("DETAILED TOKEN BREAKDOWN")
    print(f"{'='*100}\n")
    
    print(f"{'Configuration':<50} {'Input Tokens':<15} {'Output Tokens':<15} {'I/O Ratio':<10}")
    print(f"{'-'*100}")
    
    for stats in all_stats:
        print(f"{stats['file']:<50} {format_number(stats['total_input_tokens']):<15} "
              f"{format_number(stats['total_output_tokens']):<15} {stats['input_output_ratio']:.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calculate_token_costs.py <results_file.jsonl> [results_file2.jsonl ...]")
        print("\nExample:")
        print("  python calculate_token_costs.py out/history/safety_eval_200__qwen2.5-7b_N=5_R=1.jsonl \\")
        print("                                   out/history/safety_eval_200__qwen2.5-7b_N=5_R=5.jsonl")
        sys.exit(1)
    
    all_stats = []
    for jsonl_file in sys.argv[1:]:
        if not os.path.exists(jsonl_file):
            print(f"Warning: File not found: {jsonl_file}")
            continue
        
        stats = analyze_token_costs(jsonl_file)
        if stats:
            all_stats.append(stats)
    
    if not all_stats:
        print("Error: No valid result files found.")
        sys.exit(1)
    
    # Sort by number of rounds
    all_stats.sort(key=lambda x: x['num_rounds'])
    
    print_comparison(all_stats)
    
    print(f"\n{'='*100}")
    print("Note: Cost estimates use approximate API pricing ($0.003/1K input, $0.015/1K output).")
    print("Adjust rates in the script for your specific model pricing.")
    print(f"{'='*100}\n")
