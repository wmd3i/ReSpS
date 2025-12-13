import json
import argparse
from transformers import AutoTokenizer
import numpy as np
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
import warnings
warnings.filterwarnings('ignore')

def calculate_effect_size(eagle_speeds, baseline_speeds):
    """Calculate Cohen's d effect size."""
    differences = eagle_speeds - baseline_speeds
    mean_diff = np.mean(differences)
    pooled_std = np.sqrt((np.var(eagle_speeds) + np.var(baseline_speeds)) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    return mean_diff / pooled_std

def calculate_confidence_interval(eagle_speeds, baseline_speeds, confidence=0.95):
    """Calculate confidence interval for the mean difference."""
    differences = eagle_speeds - baseline_speeds
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)
    
    # t-critical value
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    # Standard error
    se = std_diff / np.sqrt(n)
    
    # Confidence interval
    margin_of_error = t_critical * se
    lower_bound = mean_diff - margin_of_error
    upper_bound = mean_diff + margin_of_error
    
    return lower_bound, upper_bound

def interpret_effect_size(cohens_d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def perform_statistical_tests(eagle_speeds, baseline_speeds, alpha=0.05):
    """Perform comprehensive statistical tests."""
    results = {}
    
    # Basic statistics
    results['n_samples'] = len(eagle_speeds)
    results['eagle_mean'] = np.mean(eagle_speeds)
    results['baseline_mean'] = np.mean(baseline_speeds)
    results['eagle_std'] = np.std(eagle_speeds, ddof=1)
    results['baseline_std'] = np.std(baseline_speeds, ddof=1)
    results['speed_ratio'] = results['eagle_mean'] / results['baseline_mean']
    
    # Paired t-test
    t_stat, p_value_t = ttest_rel(eagle_speeds, baseline_speeds)
    results['t_test'] = {
        'statistic': t_stat,
        'p_value': p_value_t,
        'significant': p_value_t < alpha,
        'test_name': 'Paired t-test'
    }
    
    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, p_value_w = wilcoxon(eagle_speeds, baseline_speeds, alternative='two-sided')
        results['wilcoxon_test'] = {
            'statistic': w_stat,
            'p_value': p_value_w,
            'significant': p_value_w < alpha,
            'test_name': 'Wilcoxon signed-rank test'
        }
    except ValueError as e:
        results['wilcoxon_test'] = {
            'error': str(e),
            'test_name': 'Wilcoxon signed-rank test'
        }
    
    # Effect size
    results['cohens_d'] = calculate_effect_size(eagle_speeds, baseline_speeds)
    
    # Confidence interval
    ci_lower, ci_upper = calculate_confidence_interval(eagle_speeds, baseline_speeds)
    results['confidence_interval'] = (ci_lower, ci_upper)
    
    # Normality test for residuals (to check t-test assumptions)
    differences = eagle_speeds - baseline_speeds
    _, p_value_normality = stats.normaltest(differences)
    results['normality_test'] = {
        'p_value': p_value_normality,
        'normal': p_value_normality > alpha,
        'test_name': 'D\'Agostino normality test'
    }
    
    return results

def format_statistical_results(results, eagle_name="EAGLE", baseline_name="Baseline"):
    """Format statistical test results for output."""
    output = []
    
    # Basic statistics
    output.append(f"Sample size: {results['n_samples']} questions")
    output.append(f"{eagle_name} mean speed: {results['eagle_mean']:.4f} tokens/second")
    output.append(f"{eagle_name} std dev: {results['eagle_std']:.4f}")
    output.append(f"{baseline_name} mean speed: {results['baseline_mean']:.4f} tokens/second")
    output.append(f"{baseline_name} std dev: {results['baseline_std']:.4f}")
    output.append(f"Speed ratio ({eagle_name}/{baseline_name}): {results['speed_ratio']:.4f}x")
    output.append("")
    
    # T-test results
    t_test = results['t_test']
    output.append("PAIRED T-TEST:")
    output.append(f"  t-statistic: {t_test['statistic']:.4f}")
    output.append(f"  p-value: {t_test['p_value']:.6f}")
    output.append(f"  Significant (α=0.05): {'YES' if t_test['significant'] else 'NO'}")
    if t_test['significant']:
        output.append("  ✓ Statistically significant difference detected")
    else:
        output.append("  ✗ No statistically significant difference detected")
    output.append("")
    
    # Wilcoxon test results
    wilcoxon_test = results['wilcoxon_test']
    output.append("WILCOXON SIGNED-RANK TEST:")
    if 'error' in wilcoxon_test:
        output.append(f"  Error: {wilcoxon_test['error']}")
    else:
        output.append(f"  W-statistic: {wilcoxon_test['statistic']:.4f}")
        output.append(f"  p-value: {wilcoxon_test['p_value']:.6f}")
        output.append(f"  Significant (α=0.05): {'YES' if wilcoxon_test['significant'] else 'NO'}")
        if wilcoxon_test['significant']:
            output.append("  ✓ Statistically significant difference detected (non-parametric)")
        else:
            output.append("  ✗ No statistically significant difference detected (non-parametric)")
    output.append("")
    
    # Effect size
    cohens_d = results['cohens_d']
    effect_interpretation = interpret_effect_size(cohens_d)
    output.append("EFFECT SIZE:")
    output.append(f"  Cohen's d: {cohens_d:.4f}")
    output.append(f"  Effect size interpretation: {effect_interpretation}")
    output.append("")
    
    # Confidence interval
    ci_lower, ci_upper = results['confidence_interval']
    output.append("CONFIDENCE INTERVAL (95%):")
    output.append(f"  Mean difference ({eagle_name} - {baseline_name}): [{ci_lower:.4f}, {ci_upper:.4f}] tokens/second")
    if ci_lower > 0:
        output.append("  ✓ Confidence interval entirely above zero - significant improvement")
    elif ci_upper < 0:
        output.append("  ✗ Confidence interval entirely below zero - significant degradation")
    else:
        output.append("  ? Confidence interval includes zero - inconclusive")
    output.append("")
    
    # Normality test
    normality_test = results['normality_test']
    output.append("ASSUMPTION CHECKING:")
    output.append(f"  Normality test p-value: {normality_test['p_value']:.6f}")
    output.append(f"  Residuals normally distributed: {'YES' if normality_test['normal'] else 'NO'}")
    if normality_test['normal']:
        output.append("  ✓ T-test assumptions met")
    else:
        output.append("  ⚠ T-test assumptions violated - consider using Wilcoxon test")
    output.append("")
    
    # Summary
    output.append("SUMMARY:")
    if t_test['significant']:
        output.append("  ✓ Statistically significant speed difference detected")
        if results['speed_ratio'] > 1:
            output.append(f"  ✓ {eagle_name} is significantly faster than {baseline_name}")
        else:
            output.append(f"  ✗ {eagle_name} is significantly slower than {baseline_name}")
    else:
        output.append("  ✗ No statistically significant speed difference detected")
        output.append("    The observed difference may be due to random variation")
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description='Calculate speed ratio between EAGLE and baseline models with statistical testing')
    parser.add_argument('--ea-file', type=str, required=True, 
                        help='Path to EAGLE evaluation JSONL file')
    parser.add_argument('--baseline-file', type=str, required=True, 
                        help='Path to baseline evaluation JSONL file')
    parser.add_argument('--tokenizer-path', type=str, 
                        default="/home/lyh/weights/hf/llama2chat/13B/",
                        help='Path to tokenizer model')
    parser.add_argument('--statistical-testing', action='store_true',
                        help='Enable comprehensive statistical significance testing')
    parser.add_argument('--eagle-name', type=str, default='EAGLE',
                        help='Name for EAGLE method in output')
    parser.add_argument('--baseline-name', type=str, default='Baseline',
                        help='Name for baseline method in output')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for statistical tests (default: 0.05)')
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    jsonl_file = args.ea_file
    jsonl_file_base = args.baseline_file
    
    # Load EAGLE data
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds = []
    eagle_question_ids = []
    for datapoint in data:
        qid = datapoint["question_id"]
        answer = datapoint["choices"][0]['turns']
        tokens = sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        speed = tokens / times if times > 0 else 0
        speeds.append(speed)
        eagle_question_ids.append(qid)

    # Load baseline data
    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    total_time = 0
    total_token = 0
    speeds0 = []
    baseline_question_ids = []
    for datapoint in data:
        qid = datapoint["question_id"]
        answer = datapoint["choices"][0]['turns']
        tokens = 0
        for i in answer:
            tokens += (len(tokenizer(i).input_ids) - 1)
        times = sum(datapoint["choices"][0]['wall_time'])
        speed = tokens / times if times > 0 else 0
        speeds0.append(speed)
        baseline_question_ids.append(qid)
        total_time += times
        total_token += tokens

    # Ensure we have matching question IDs for statistical testing
    eagle_dict = dict(zip(eagle_question_ids, speeds))
    baseline_dict = dict(zip(baseline_question_ids, speeds0))
    
    # Find common question IDs
    common_ids = sorted(set(eagle_question_ids) & set(baseline_question_ids))
    
    if len(common_ids) == 0:
        print("Warning: No matching question IDs found between files")
        print("Statistical testing requires matching question IDs")
        eagle_speeds_matched = speeds
        baseline_speeds_matched = speeds0
    else:
        # Extract speeds for common questions
        eagle_speeds_matched = [eagle_dict[qid] for qid in common_ids]
        baseline_speeds_matched = [baseline_dict[qid] for qid in common_ids]

    # Calculate basic results
    eagle_speed = np.array(speeds).mean()
    baseline_speed = np.array(speeds0).mean()
    speed_ratio = eagle_speed / baseline_speed
    
    print(f"{args.eagle_name} average speed: {eagle_speed:.4f} tokens/second")
    print(f"{args.baseline_name} average speed: {baseline_speed:.4f} tokens/second")
    print(f"Speed ratio ({args.eagle_name}/{args.baseline_name}): {speed_ratio:.4f}x faster")
    
    if args.statistical_testing:
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*60)
        
        if len(common_ids) == 0:
            print("❌ Cannot perform statistical testing: No matching question IDs")
            print("   Statistical tests require paired data (same questions for both methods)")
        else:
            print(f"✅ Using {len(common_ids)} matched questions for statistical testing")
            
            # Perform statistical tests
            results = perform_statistical_tests(
                np.array(eagle_speeds_matched), 
                np.array(baseline_speeds_matched), 
                alpha=args.alpha
            )
            
            # Format and print results
            statistical_output = format_statistical_results(
                results, 
                eagle_name=args.eagle_name, 
                baseline_name=args.baseline_name
            )
            print(statistical_output)

if __name__ == "__main__":
    main()
