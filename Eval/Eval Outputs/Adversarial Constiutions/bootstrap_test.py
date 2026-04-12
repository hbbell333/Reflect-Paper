import numpy as np
import json
import re
import os
import sys

def extract_multi_objective_scores(text):
    scores = []
    lines = text.strip().split('\n')
    for line in lines:
        match = re.search(r'(?:--|:)\s*(\d+)', line)
        if match:
            scores.append(int(match.group(1)))
    return scores

def load_scores_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    scores = []
    for entry in data[1:]:
        if len(entry) >= 2:
            eval_text = entry[1]
            individual_scores = extract_multi_objective_scores(eval_text)
            if individual_scores:
                avg_score = np.mean(individual_scores)
                scores.append(avg_score)
    
    return np.array(scores)

def load_violation_proportions_from_recoded(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    violation_proportions = []
    for entry in data[1:]:
        if len(entry) >= 2:
            recoded_scores = entry[1]
            if recoded_scores:
                num_violations = sum(1 for score in recoded_scores if score == 0)
                total_principles = len(recoded_scores)
                violation_prop = num_violations / total_principles if total_principles > 0 else 0.0
                violation_proportions.append(violation_prop)
    
    return np.array(violation_proportions)

def paired_bootstrap_test(baseline, comparison, n_boot=10000, ci=95, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    baseline = np.array(baseline)
    comparison = np.array(comparison)
    
    if len(baseline) != len(comparison):
        raise ValueError("Arrays must be the same length")
    
    observed_diff = np.mean(comparison - baseline)
    
    diffs = []
    n = len(baseline)
    
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        diff = np.mean(comparison[idx] - baseline[idx])
        diffs.append(diff)
    
    diffs = np.array(diffs)
    
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    ci_lower = np.percentile(diffs, lower_pct)
    ci_upper = np.percentile(diffs, upper_pct)
    
    p_value = np.mean(diffs <= 0)
    
    return {
        "mean_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value
    }

def is_recoded_file(filepath):
    filename = os.path.basename(filepath)
    return 'RECODED' in filename.upper()

def run_bootstrap_test(baseline_path, comparison_path, seed=42):
    is_recoded = is_recoded_file(baseline_path) or is_recoded_file(comparison_path)
    
    if is_recoded:
        baseline_scores = load_violation_proportions_from_recoded(baseline_path)
        comparison_scores = load_violation_proportions_from_recoded(comparison_path)
        test_type = "violation proportions"
        min_len = min(len(baseline_scores), len(comparison_scores))
        baseline_scores = baseline_scores[:min_len]
        comparison_scores = comparison_scores[:min_len]
        result = paired_bootstrap_test(comparison_scores, baseline_scores, seed=seed)
        mean_diff_label = "baseline - comparison"
    else:
        baseline_scores = load_scores_from_json(baseline_path)
        comparison_scores = load_scores_from_json(comparison_path)
        test_type = "average scores"
        min_len = min(len(baseline_scores), len(comparison_scores))
        baseline_scores = baseline_scores[:min_len]
        comparison_scores = comparison_scores[:min_len]
        result = paired_bootstrap_test(baseline_scores, comparison_scores, seed=seed)
        mean_diff_label = "comparison - baseline"
    
    baseline_name = os.path.basename(baseline_path)
    comparison_name = os.path.basename(comparison_path)
    
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append(f"\nTest Type: {test_type}")
    output_lines.append(f"Baseline File: {baseline_name}")
    output_lines.append(f"Comparison File: {comparison_name}")
    output_lines.append(f"\nLoaded {len(baseline_scores)} paired scores")
    
    if is_recoded:
        output_lines.append(f"Baseline mean: {np.mean(baseline_scores) * 100:.4f}%")
        output_lines.append(f"Comparison mean: {np.mean(comparison_scores) * 100:.4f}%")
        output_lines.append(f"\nBootstrap Test Results:")
        output_lines.append(f"  Mean difference ({mean_diff_label}): {result['mean_diff'] * 100:.4f}%")
        output_lines.append(f"  95% CI: [{result['ci_lower'] * 100:.4f}%, {result['ci_upper'] * 100:.4f}%]")
        output_lines.append(f"  p-value: {result['p_value']:.4f}")
        
        if result['mean_diff'] > 0:
            output_lines.append(f"Comparison has {result['mean_diff'] * 100:.4f}% fewer violations on average (better)")
        elif result['mean_diff'] < 0:
            output_lines.append(f"Comparison has {abs(result['mean_diff']) * 100:.4f}% more violations on average (worse)")
        else:
            output_lines.append(f"Comparison and baseline have the same violation rate on average")
    else:
        output_lines.append(f"Baseline mean: {np.mean(baseline_scores):.4f}")
        output_lines.append(f"Comparison mean: {np.mean(comparison_scores):.4f}")
        output_lines.append(f"\nBootstrap Test Results:")
        output_lines.append(f"  Mean difference ({mean_diff_label}): {result['mean_diff']:.4f}")
        output_lines.append(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        output_lines.append(f"  p-value: {result['p_value']:.4f}")
        if result['mean_diff'] > 0:
            output_lines.append(f"Comparison is {result['mean_diff']:.4f} points better on average")
        elif result['mean_diff'] < 0:
            output_lines.append(f"Comparison is {abs(result['mean_diff']):.4f} points worse on average")
        else:
            output_lines.append(f"Comparison and baseline have the same mean score")
    
    return "\n".join(output_lines)

if __name__ == "__main__":
    # file_paths = [
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/ICL/EVAL_const2_gpt4.1mini_safeRLHF.json',
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/EVAL_NEW_efficient_GPT_safeRLHF_const2_20251018_173052.json',

    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/ICL/EVAL_const2_claude_haiku3.5_safeRLHF_ICL.json',
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/EVAL_NEW_efficient_CLAUDE_safeRLHF_const2_20251018_173618.json',

    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/ICL/EVAL_base_mistral_safeRLHF_ICL_20251221_172546.json',
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/EVAL_safeRLHF_MISTRAL_reflect_results_20251220_212044_20251221_172057.json',

    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/ICL/EVAL_const2_gpt4.1mini_anthharm_ICL.json',
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/Principle Select/EVAL_NEW_efficient_GPT_anthharm_const2_20251018_173459.json',

    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/ICL/EVAL_const2_claude_haiku3.5_anthharm_ICL.json',
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/Principle Select/EVAL_NEW_efficient_CLAUDE_anthharm_const2_20251018_173657.json',

    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/ICL/EVAL_mistral_anthharm_ICL_20260110_110021.json',
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/Principle Select/EVAL_MISTRAL_reflect_anthharm_results_20251223_051510_20251223_133013.json', 

    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/ICL/RECODED_EVAL_const2_gpt4.1mini_safeRLHF.json',
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/RECODED_EVAL_NEW_efficient_GPT_safeRLHF_const2_20251018_173052.json',

    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/ICL/RECODED_EVAL_const2_claude_haiku3.5_safeRLHF_ICL.json',
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/RECODED_EVAL_NEW_efficient_CLAUDE_safeRLHF_const2_20251018_173618.json',

    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/ICL/RECODED_EVAL_base_mistral_safeRLHF_ICL_20251221_172546.json',
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/SafeRLHF Const 2/Principle Select/RECODED_EVAL_safeRLHF_MISTRAL_reflect_results_20251220_212044_20251221_172057.json',

    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/ICL/RECODED_EVAL_const2_gpt4.1mini_anthharm_ICL.json',
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/Principle Select/RECODED_EVAL_NEW_efficient_GPT_anthharm_const2_20251018_173459.json',

    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/ICL/RECODED_EVAL_const2_claude_haiku3.5_anthharm_ICL.json',
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/Principle Select/RECODED_EVAL_NEW_efficient_CLAUDE_anthharm_const2_20251018_173657.json',

    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/ICL/RECODED_EVAL_mistral_anthharm_ICL_20260110_110021.json',
    #     '/Users/carolinezhang/Desktop/Reflect/MO_Evaluation/Eval Outputs/Adversarial Constiutions/Anthharm Const 2/Principle Select/RECODED_EVAL_MISTRAL_reflect_anthharm_results_20251223_051510_20251223_133013.json'
    # ]
    
    # file_paths = [
    #     "/Users/carolinezhang/Desktop/Reflect/GPT-SFT/noICL/EVAL_gpt_base_no_ICL_20260111_215830.json",
    #     "/Users/carolinezhang/Desktop/Reflect/GPT-SFT/noICL/EVAL_New_DPO_GPT_results_noICL_20260111_203859.json",

    #     "/Users/carolinezhang/Desktop/Reflect/GPT-SFT/noICL/EVAL_gpt_base_no_ICL_20260111_215830.json",
    #     "/Users/carolinezhang/Desktop/Reflect/GPT-SFT/noICL/EVAL_gpt_SFT_2_no_ICL_20260111_214348.json"
    # ]

    file_paths = [
        r'C:\Users\hbbel\OneDrive\Desktop\Research\Git Repos\Reflect\MO_Evaluation\Eval Outputs\Adversarial Constiutions\SafeRLHF Const 2\ICL\EVAL_const2_gpt4.1mini_safeRLHF.json',
        r'C:\Users\hbbel\OneDrive\Desktop\Research\Git Repos\Reflect\Evaluation\pairwise-eval\gpt 4.1 rerun\EVAL_rerun_GPT_extra_tokens_safeRLHF_20260405_130521_20260405_151440.json'
    ]

    output_path = "pvals_gpt_safeRLHF_rerun.txt"
    
    all_results = []
    
    for i in range(0, len(file_paths), 2):
        if i + 1 < len(file_paths):
            baseline_path = file_paths[i]
            comparison_path = file_paths[i + 1]
            result_text = run_bootstrap_test(baseline_path, comparison_path)
            all_results.append(result_text)
            all_results.append("")
    
    output_text = "\n".join(all_results)
    
    with open(output_path, 'w') as f:
        f.write(output_text)