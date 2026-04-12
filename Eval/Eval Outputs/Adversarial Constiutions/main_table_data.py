import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

def extract_principle_score(text):
    # Match both formats: "[Principle] -- 5" and "[Principle]: 5"
    match = re.search(r'(?:--|]:)\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None

def extract_principle_name(text):
    match = re.search(r'\[([^\]]+)\]', text)
    if match:
        return match.group(1)
    return None

def process_evaluation_file(file_path):
    """
    Process a single evaluation JSON file and extract all principle scores.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Use dict to store scores for each principle
    principle_scores = defaultdict(list)
    
    # Skip the first entry 
    for entry in data[1:]:
        if len(entry) >= 2:
            evaluation_string = entry[1]
            # Can split by newlines
            lines = evaluation_string.split('\n')
            for line in lines:
                line = line.strip()
                if line and ('--' in line or ']:' in line):
                    principle_name = extract_principle_name(line)
                    score = extract_principle_score(line)
                    
                    if principle_name and score is not None:
                        principle_scores[principle_name].append(score)
    
    return principle_scores

def calculate_mean_and_moe(scores):
    """
    Calculate mean and 95% MOE
    MoE = 1.96 * (sigma / root n)
    """
    if len(scores) == 0:
        return None, None
    
    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    std = np.std(scores_array, ddof=1)  
    n = len(scores)
    
    if n > 1:
        moe = 1.96 * (std / np.sqrt(n))
    else:
        moe = 0.0
    
    return mean, moe

def print_results(principle_scores):
    """
    Print results for each principle showing name, average score, and MOE.
    Returns a formatted string for saving to file and overall stats.
    """
    #sorted_principles = sorted(principle_scores.keys())
    lines = []
    
    lines.append(f"{'Principle':<40} {'Mean Score':<15} {'95% MOE':<15} {'N':<10}")
    lines.append("=" * 80)
    
    # Collect all scores across all principles for overall statistics
    all_scores = []
    
    for principle in principle_scores.keys():
        scores = principle_scores[principle]
        mean, moe = calculate_mean_and_moe(scores)
        
        if mean is not None:
            line = f"{principle:<40} {mean:<15.3f} {moe:<15.3f} {len(scores):<10}"
            lines.append(line)
            print(line)
            all_scores.extend(scores)
    
    # Calculate overall mean and MOE 
    overall_mean, overall_moe = calculate_mean_and_moe(all_scores)
    
    return "\n".join(lines), overall_mean, overall_moe

#file_path = '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/EVAL_NEW_efficient_GPT_safeRLHF_const2_20251018_173052.json'

file_paths = [
   '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/ICL/EVAL_const2_gpt4.1mini_safeRLHF.json',
   '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/ICL/EVAL_const2_claude_haiku3.5_safeRLHF_ICL.json',
   '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/ICL/EVAL_base_mistral_safeRLHF_ICL_20251221_172546.json',
   '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/EVAL_NEW_efficient_GPT_safeRLHF_const2_20251018_173052.json',
   '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/EVAL_NEW_efficient_CLAUDE_safeRLHF_const2_20251018_173618.json',
   '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/EVAL_safeRLHF_MISTRAL_reflect_results_20251220_212044_20251221_172057.json',
   '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/ICL/EVAL_const2_gpt4.1mini_anthharm_ICL.json',
   '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/ICL/EVAL_const2_claude_haiku3.5_anthharm_ICL.json',
   '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/ICL/EVAL_mistral_anthharm_ICL_20260110_110021.json',
   '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/Principle Select/EVAL_NEW_efficient_GPT_anthharm_const2_20251018_173459.json',
   '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/Principle Select/EVAL_NEW_efficient_CLAUDE_anthharm_const2_20251018_173657.json',
   '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/Principle Select/EVAL_MISTRAL_reflect_anthharm_results_20251223_051510_20251223_133013.json'
]

# size ablation file paths
# file_paths = [
#     '/Users/carolinezhang/Desktop/Reflect/Size Ablation/EVAL_GPT_nano_ICL_base_responses_20251227_103817.json',
#     '/Users/carolinezhang/Desktop/Reflect/Size Ablation/EVAL_Fixed_nano_results_20251230_090859.json',
#     '/Users/carolinezhang/Desktop/Reflect/Size Ablation/EVAL_gpt4.1_safeRLHF_ICL_20251229_204417.json', 
#     '/Users/carolinezhang/Desktop/Reflect/Size Ablation/EVAL_BIGGPT_safeRLHF_20251229_191430_20251229_204008.json'
# ]

# self finetuning file paths
# file_paths = [
#     "/Users/carolinezhang/Desktop/Reflect/GPT-SFT/noICL/EVAL_gpt_base_no_ICL_20260111_215830.json",
#     "/Users/carolinezhang/Desktop/Reflect/GPT-SFT/noICL/EVAL_New_DPO_GPT_results_noICL_20260111_203859.json",
#     "/Users/carolinezhang/Desktop/Reflect/GPT-SFT/noICL/EVAL_gpt_SFT_2_no_ICL_20260111_214348.json"
# ]

# # safety evaluation file paths
# file_paths = [
#     "/Users/carolinezhang/Desktop/Reflect/Safety Evaluation/EVAL_harmful_saferlhf500_reflect_responses_20260319_130541.json", 
#     "/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/HHH/GPT/EVAL_gpt_HHH_icl_20260108_114426.json"
# ]

# Gpt rerun file paths
file_paths = [
        r'C:\Users\hbbel\OneDrive\Desktop\Research\Git Repos\Reflect\MO_Evaluation\Eval Outputs\Adversarial Constiutions\SafeRLHF Const 2\ICL\EVAL_const2_gpt4.1mini_safeRLHF.json',
        r'C:\Users\hbbel\OneDrive\Desktop\Research\Git Repos\Reflect\Evaluation\pairwise-eval\gpt 4.1 rerun\EVAL_rerun_GPT_extra_tokens_safeRLHF_20260405_130521_20260405_151440.json'
    ]

# safety evaluation file paths
# file_paths = [
#     "/Users/carolinezhang/Desktop/Reflect/Safety Evaluation/EVAL_harmful_saferlhf500_reflect_responses_20260319_130541.json", 
#     "/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/HHH/GPT/EVAL_gpt_HHH_icl_20260108_114426.json"
# ]
# Process all files and save results
output_lines = []

for file_path in file_paths:
    print(f"\nProcessing: {file_path}")
    
    filename = Path(file_path).name
    
    try:
        principle_scores = process_evaluation_file(file_path)
        
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"File: {filename}")
        output_lines.append(f"{'='*80}\n")
        
        # Get formatted results and overall stats
        results_text, overall_mean, overall_moe = print_results(principle_scores)
        output_lines.append(results_text)
        output_lines.append("")
        
        # Add overall summary
        if overall_mean is not None and overall_moe is not None:
            summary_line = f"Overall Average (across all principles): {overall_mean:.3f} ± {overall_moe:.3f} (95% MOE)"
            output_lines.append(summary_line)
            output_lines.append("")
            print(f"Overall Average: {overall_mean:.3f} ± {overall_moe:.3f} (95% MOE)")  
        
    except Exception as e:
        error_msg = f"Error processing {filename}: {str(e)}"
        print(error_msg)
        output_lines.append(error_msg)
        output_lines.append("")

# Save all results
output_file = Path(__file__).parent / "gpt_rerun_avg_results.txt"
output_file = Path(__file__).parent / "updated_main_table_data.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(output_lines))

print(f"\n\nResults saved to: {output_file}")


