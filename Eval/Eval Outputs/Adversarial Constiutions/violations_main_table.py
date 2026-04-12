import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

def process_violation_file(file_path):
    """
    Process a single recoded evaluation JSON file.
    Returns a dictionary mapping principle indices to violation percentages.
    Also return total count.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 0 = violation, 1 = conformance
    principle_violations = defaultdict(lambda: {'violations': 0, 'total': 0})
    total_violations = 0
    total_evaluations = 0
    
    # Skip the first entry
    for entry in data[1:]:
        if len(entry) >= 2:
            violation_array = entry[1]  
            
            # Process each principle's violation status
            for principle_idx, value in enumerate(violation_array):
                principle_violations[principle_idx]['total'] += 1
                total_evaluations += 1
                if value == 0: 
                    principle_violations[principle_idx]['violations'] += 1
                    total_violations += 1
    
    return principle_violations, total_violations, total_evaluations

def calculate_binomial_moe(violations, total):
    """
    Calculate 95% MOE for a binomial proportion.
    MOE = 1.96 * sqrt(p(1-p)/n)
    p = porportion, n = total count
    """
    if total == 0:
        return None, None
    
    p = violations / total  # proportion (0-1)
    n = total
    
    # Standard error for binomial proportion
    se = np.sqrt(p * (1 - p) / n)
    moe = 1.96 * se
    
    return p, moe

def print_results(principle_violations, total_violations, total_evaluations):
    """
    Print violation percentage for each principle and calculate overall binomial proportion.
    Returns formatted string and overall stats using binomial MOE.
    """
    lines = []
    
    lines.append(f"{'Principle Index':<20} {'Violation %':<15} {'95% MOE':<15}")
    lines.append("=" * 62)
    
    for principle_idx in sorted(principle_violations.keys()):
        counts = principle_violations[principle_idx]
        p, moe = calculate_binomial_moe(counts['violations'], counts['total'])
        violation_percent = p * 100 if p is not None else 0.0
        moe_percent = moe * 100 if moe is not None else 0.0
        line = f"{principle_idx:<20} {violation_percent:<15.3f} ± {moe_percent:.3f}%"
        lines.append(line)
        print(line)
    
    # Calculate overall violation proportion and MOE using binomial formula
    overall_proportion, overall_moe = calculate_binomial_moe(total_violations, total_evaluations)
    
    # Convert to percentages
    overall_mean = overall_proportion * 100 if overall_proportion is not None else None
    overall_moe_percent = overall_moe * 100 if overall_moe is not None else None
    
    return "\n".join(lines), overall_mean, overall_moe_percent, total_evaluations

# File paths to process (RECODED files)
# file_paths = [
#     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/ICL/RECODED_EVAL_const2_gpt4.1mini_safeRLHF.json',
#     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/ICL/RECODED_EVAL_const2_claude_haiku3.5_safeRLHF_ICL.json',
#     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/ICL/RECODED_EVAL_base_mistral_safeRLHF_ICL_20251221_172546.json',
#     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/RECODED_EVAL_NEW_efficient_GPT_safeRLHF_const2_20251018_173052.json',
#     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/RECODED_EVAL_NEW_efficient_CLAUDE_safeRLHF_const2_20251018_173618.json',
#     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/RECODED_EVAL_safeRLHF_MISTRAL_reflect_results_20251220_212044_20251221_172057.json',
#     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/ICL/RECODED_EVAL_const2_gpt4.1mini_anthharm_ICL.json',
#     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/ICL/RECODED_EVAL_const2_claude_haiku3.5_anthharm_ICL.json',
#     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/ICL/RECODED_EVAL_mistral_anthharm_ICL_20260110_110021.json',
#     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/Principle Select/RECODED_EVAL_NEW_efficient_GPT_anthharm_const2_20251018_173459.json',
#     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/Principle Select/RECODED_EVAL_NEW_efficient_CLAUDE_anthharm_const2_20251018_173657.json',
#     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/Principle Select/RECODED_EVAL_MISTRAL_reflect_anthharm_results_20251223_051510_20251223_133013.json',
# ]

#self refine comparison file paths
# Gpt rerun file paths
file_paths = [
        r'C:\Users\hbbel\OneDrive\Desktop\Research\Git Repos\Reflect\MO_Evaluation\Eval Outputs\Adversarial Constiutions\SafeRLHF Const 2\ICL\RECODED_EVAL_const2_gpt4.1mini_safeRLHF.json',
        r'C:\Users\hbbel\OneDrive\Desktop\Research\Git Repos\Reflect\Evaluation\pairwise-eval\gpt 4.1 rerun\RECODED_EVAL_rerun_GPT_extra_tokens_safeRLHF_20260405_130521_20260405_151440.json'
    ]

output_lines = []

for file_path in file_paths:
    print(f"\nProcessing: {file_path}")
    
    filename = Path(file_path).name
    
    try:
        principle_violations, total_violations, total_evaluations = process_violation_file(file_path)
        
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"File: {filename}")
        output_lines.append(f"{'='*80}\n")
        
        results_text, overall_mean, overall_moe, n = print_results(principle_violations, total_violations, total_evaluations)
        output_lines.append(results_text)
        output_lines.append("")
        
        if overall_mean is not None and overall_moe is not None:
            summary_line = f"Overall Violation Rate (pooled across all principles): {overall_mean:.3f}% ± {overall_moe:.3f}% (95% MOE, n={n})"
            output_lines.append(summary_line)
            output_lines.append("")
            print(f"Overall Violation Rate: {overall_mean:.3f}% ± {overall_moe:.3f}% (95% MOE, n={n})")
        
    except Exception as e:
        error_msg = f"Error processing {filename}: {str(e)}"
        print(error_msg)
        output_lines.append(error_msg)
        output_lines.append("")

# Save all results
output_file = Path(__file__).parent / "gpt_rerun_CI_results.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))

print(f"\n\nResults saved to: {output_file}")