#!/usr/bin/env python3
"""
Category Aggregator: Analyze per-category accuracy from debate results.
Usage: python analyze_categories.py <results_file.jsonl>
"""
import json
import sys
from collections import defaultdict
import numpy as np


def analyze_categories(jsonl_file):
    """Analyze per-category accuracy across all rounds from saved results."""
    
    # Load results
    results = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    print(f"\n{'='*70}")
    print(f"Category Analysis: {jsonl_file}")
    print(f"Total Samples: {len(results)}")
    print(f"{'='*70}\n")
    
    # Extract all rounds
    num_rounds = len(results[0]) if results else 0
    
    # Aggregate by category for each round
    for round_idx in range(num_rounds):
        round_key = str(round_idx)
        cat_correct = defaultdict(list)
        cat_flips = defaultdict(int)
        
        for sample in results:
            if round_key not in sample:
                continue
            
            round_data = sample[round_key]
            category = round_data.get('category', 'unknown')
            is_correct = round_data.get('debate_answer_iscorr', False)
            cat_correct[category].append(int(is_correct))
            
            # Track flips (if agent changed from correct to incorrect or vice versa)
            if round_idx > 0:
                prev_correct = sample[str(round_idx-1)].get('debate_answer_iscorr', False)
                if prev_correct != is_correct:
                    cat_flips[category] += 1
        
        # Print round statistics
        print(f"Round {round_idx}:")
        print(f"  {'Category':<45} {'Accuracy':<12} {'Count':<8} {'Flips':<8}")
        print(f"  {'-'*75}")
        
        overall_correct = []
        for cat in sorted(cat_correct.keys()):
            corrects = cat_correct[cat]
            acc = np.mean(corrects) if corrects else 0.0
            flips = cat_flips.get(cat, 0) if round_idx > 0 else 0
            print(f"  {cat:<45} {acc:>6.1%}       {len(corrects):<8} {flips:<8}")
            overall_correct.extend(corrects)
        
        overall_acc = np.mean(overall_correct) if overall_correct else 0.0
        print(f"  {'-'*75}")
        print(f"  {'OVERALL':<45} {overall_acc:>6.1%}       {len(overall_correct):<8}\n")
    
    # Additional analysis: Most improved/degraded categories
    if num_rounds > 1:
        print(f"\n{'='*70}")
        print("Category Improvement (First vs Last Round)")
        print(f"{'='*70}\n")
        
        first_round_acc = defaultdict(list)
        last_round_acc = defaultdict(list)
        
        for sample in results:
            first_data = sample['0']
            last_data = sample[str(num_rounds-1)]
            category = first_data.get('category', 'unknown')
            
            first_round_acc[category].append(int(first_data.get('debate_answer_iscorr', False)))
            last_round_acc[category].append(int(last_data.get('debate_answer_iscorr', False)))
        
        improvements = []
        for cat in sorted(first_round_acc.keys()):
            first_acc = np.mean(first_round_acc[cat]) if first_round_acc[cat] else 0.0
            last_acc = np.mean(last_round_acc[cat]) if last_round_acc[cat] else 0.0
            delta = last_acc - first_acc
            improvements.append((cat, first_acc, last_acc, delta, len(first_round_acc[cat])))
        
        # Sort by improvement
        improvements.sort(key=lambda x: x[3], reverse=True)
        
        print(f"  {'Category':<45} {'First':<10} {'Last':<10} {'Change':<10} {'Count':<8}")
        print(f"  {'-'*85}")
        for cat, first, last, delta, count in improvements:
            delta_str = f"{delta:+.1%}"
            print(f"  {cat:<45} {first:>6.1%}     {last:>6.1%}     {delta_str:>8}   {count:<8}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_categories.py <results_file.jsonl>")
        print("Example: python analyze_categories.py out/history/safety_eval_200__qwen2.5-7b_N=5_R=5.jsonl")
        sys.exit(1)
    
    analyze_categories(sys.argv[1])
