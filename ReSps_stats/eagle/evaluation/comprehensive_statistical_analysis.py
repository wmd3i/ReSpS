#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis for EAGLE Speed Comparisons

This script performs advanced statistical analysis across multiple benchmarks
and methods, including:
- Multiple comparison corrections (Bonferroni, FDR)
- Effect size analysis across benchmarks (aggregated)
- Visualization of statistical results
- Concise reporting of p-values per method pair

Usage:
    python -m eagle.evaluation.comprehensive_statistical_analysis \
        --results-dir log/20250727_0412_optimized_ppo \
        --tokenizer-path meta-llama/Llama-3.1-8B-Instruct \
        --output-dir statistical_analysis_results
        
Example:
    # Generate analysis for LLaMA 3.1-8B
    python -m eagle.evaluation.comprehensive_statistical_analysis \
        --results-dir stats_test_data/results_llama3.1-8B/20250729_184423_ofl128_llama318b \
        --tokenizer-path meta-llama/Llama-3.1-8B-Instruct
    # Generate analysis for Vicuna 13B
    python -m eagle.evaluation.comprehensive_statistical_analysis \
        --results-dir stats_test_data/results_vicuna-13b/20250730_160223_ofl128_vicuna13b \
        --tokenizer-path lmsys/vicuna-13b-v1.5
    or (the above is used in the paper):
    python -m eagle.evaluation.comprehensive_statistical_analysis \
        --results-dir stats_test_data/results_vicuna-13b/20250730_160223_ofl128_vicuna13b \
        --tokenizer-path lmsys/vicuna-13b-v1.3
    # Generate analysis for LLaMA 3.3-70B
    python -m eagle.evaluation.comprehensive_statistical_analysis \
        --results-dir stats_test_data/results_llama3.3_70b/20250731_052920_ofl128_llama3370b \
        --tokenizer-path meta-llama/Llama-3.3-70B-Instruct
"""

import json
import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveStatisticalAnalyzer:
    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False)
        # Structure: combo_name -> {'ALL': [comparison_dict]}
        self.results = defaultdict(dict)
        # Extract model identifier from tokenizer path
        self.model_identifier = self._extract_model_identifier(tokenizer_path)

    def _extract_model_identifier(self, tokenizer_path):
        """Extract model identifier from tokenizer path for baseline file naming."""
        if "llama-3.1-8b" in tokenizer_path.lower():
            return "LLaMA3.1-8B"
        elif "llama-3.3-70b" in tokenizer_path.lower():
            return "LLaMA3.3-70B"
        elif "vicuna-13b" in tokenizer_path.lower():
            return "Vicuna-13B"
        else:
            # Fallback: try to extract from path
            path_parts = tokenizer_path.lower().split('/')
            if any("llama" in part and "3.1" in part and "8b" in part for part in path_parts):
                return "LLaMA3.1-8B"
            elif any("llama" in part and "3.3" in part and "70b" in part for part in path_parts):
                return "LLaMA3.3-70B"
            elif any("vicuna" in part and "13b" in part for part in path_parts):
                return "Vicuna-13B"
            else:
                # Default fallback
                return "LLaMA3.1-8B"

    def load_speed_data(self, file_path):
        """Load speed data from a JSONL file."""
        speeds = []
        question_ids = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                qid = json_obj["question_id"]
                tokens = sum(json_obj["choices"][0]['new_tokens'])
                times = sum(json_obj["choices"][0]['wall_time'])
                speed = tokens / times if times > 0 else 0
                speeds.append(speed)
                question_ids.append(qid)

        return np.array(speeds), question_ids

    def calculate_effect_size(self, method1_speeds, method2_speeds):
        """Calculate Cohen's d effect size."""
        differences = method1_speeds - method2_speeds
        mean_diff = np.mean(differences)
        pooled_std = np.sqrt((np.var(method1_speeds) + np.var(method2_speeds)) / 2)
        if pooled_std == 0:
            return 0.0
        return mean_diff / pooled_std

    def aggregate_comparisons_across_benchmarks(self, results_dir, benchmarks):
        """
        Aggregate all matched speed samples across benchmarks for each method pair
        and run one statistical test per combination.
        """
        results_dir = Path(results_dir)
        policy_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('optimized_')]

        aggregated = defaultdict(lambda: {'speeds1': [], 'speeds2': [], 'ids': set()})

        for policy_dir in policy_dirs:
            policy_name = policy_dir.name
            for benchmark in benchmarks:
                policy_file = policy_dir / "evaluation" / f"{benchmark}_results.jsonl"
                eagle3_file = policy_dir / "baseline_results" / f"{benchmark}_{self.model_identifier}_eagle3.jsonl"
                baseline_file = policy_dir / "baseline_results" / f"{benchmark}_{self.model_identifier}_baseline.jsonl"

                pairs = []
                if policy_file.exists() and eagle3_file.exists():
                    pairs.append((policy_file, eagle3_file, policy_name, "EAGLE3"))
                if policy_file.exists() and baseline_file.exists():
                    pairs.append((policy_file, baseline_file, policy_name, "Baseline"))
                if eagle3_file.exists() and baseline_file.exists():
                    pairs.append((eagle3_file, baseline_file, "EAGLE3", "Baseline"))

                for file1, file2, name1, name2 in pairs:
                    speeds1, ids1 = self.load_speed_data(file1)
                    speeds2, ids2 = self.load_speed_data(file2)
                    dict1 = dict(zip(ids1, speeds1))
                    dict2 = dict(zip(ids2, speeds2))
                    common_ids = sorted(set(ids1) & set(ids2))
                    if not common_ids:
                        continue
                    key = f"{name1}_vs_{name2}"
                    aggregated[key]['speeds1'].extend([dict1[qid] for qid in common_ids])
                    aggregated[key]['speeds2'].extend([dict2[qid] for qid in common_ids])
                    aggregated[key]['ids'].update(common_ids)

        # Run aggregated tests
        for combo_name, data in aggregated.items():
            speeds1 = np.array(data['speeds1'])
            speeds2 = np.array(data['speeds2'])
            if len(speeds1) == 0 or len(speeds2) == 0:
                continue

            # Paired tests
            t_stat, t_p = ttest_rel(speeds1, speeds2)
            try:
                w_stat, w_p = wilcoxon(speeds1, speeds2, alternative='two-sided')
                wilcoxon_success = True
            except ValueError:
                w_stat, w_p = np.nan, np.nan
                wilcoxon_success = False

            cohens_d = self.calculate_effect_size(speeds1, speeds2)
            diffs = speeds1 - speeds2
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)
            n = len(diffs)
            t_critical = stats.t.ppf(0.975, df=n - 1)
            se = std_diff / np.sqrt(n)
            ci_lower = mean_diff - t_critical * se
            ci_upper = mean_diff + t_critical * se

            method1, method2 = combo_name.split("_vs_")
            self.results[combo_name]['ALL'] = [{
                'benchmark': 'ALL',
                'method1': method1,
                'method2': method2,
                'n_samples': n,
                'method1_mean': np.mean(speeds1),
                'method2_mean': np.mean(speeds2),
                'method1_std': np.std(speeds1, ddof=1),
                'method2_std': np.std(speeds2, ddof=1),
                'speed_ratio': np.mean(speeds1) / np.mean(speeds2) if np.mean(speeds2) != 0 else np.nan,
                't_statistic': t_stat,
                't_p_value': t_p,
                'wilcoxon_statistic': w_stat if wilcoxon_success else np.nan,
                'wilcoxon_p_value': w_p if wilcoxon_success else np.nan,
                'cohens_d': cohens_d,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'mean_difference': mean_diff,
                'significant_t_test': t_p < 0.05,
                'significant_wilcoxon': (w_p < 0.05) if wilcoxon_success else False
            }]

    def apply_multiple_comparison_corrections(self):
        """Apply multiple comparison corrections (Bonferroni & FDR) to aggregated combos."""
        # Collect all t and Wilcoxon p-values and keep references
        t_p_values = []
        w_p_values = []
        t_comps = []
        w_comps = []

        for combo_name, benchmark_results in self.results.items():
            comp = benchmark_results.get('ALL', [None])[0]
            if comp is None:
                continue
            t_p_values.append(comp['t_p_value'])
            t_comps.append(comp)
            if not np.isnan(comp.get('wilcoxon_p_value', np.nan)):
                w_p_values.append(comp['wilcoxon_p_value'])
                w_comps.append(comp)

        # Corrections for t-test
        if t_p_values:
            p_vals = np.array(t_p_values)
            bonf = np.minimum(p_vals * len(p_vals), 1.0)
            sorted_idx = np.argsort(p_vals)
            fdr = np.zeros_like(p_vals)
            for i, idx in enumerate(sorted_idx):
                fdr[idx] = p_vals[idx] * len(p_vals) / (i + 1)
            fdr = np.minimum(fdr, 1.0)
            for i, comp in enumerate(t_comps):
                comp['bonferroni_p_value'] = bonf[i]
                comp['fdr_p_value'] = fdr[i]
                comp['significant_bonferroni'] = bonf[i] < 0.05
                comp['significant_fdr'] = fdr[i] < 0.05

        # Corrections for Wilcoxon
        if w_p_values:
            p_vals_w = np.array(w_p_values)
            bonf_w = np.minimum(p_vals_w * len(p_vals_w), 1.0)
            sorted_idx_w = np.argsort(p_vals_w)
            fdr_w = np.zeros_like(p_vals_w)
            for i, idx in enumerate(sorted_idx_w):
                fdr_w[idx] = p_vals_w[idx] * len(p_vals_w) / (i + 1)
            fdr_w = np.minimum(fdr_w, 1.0)
            for i, comp in enumerate(w_comps):
                comp['bonferroni_p_value_wilcoxon'] = bonf_w[i]
                comp['fdr_p_value_wilcoxon'] = fdr_w[i]
                comp['significant_bonferroni_wilcoxon'] = bonf_w[i] < 0.05
                comp['significant_fdr_wilcoxon'] = fdr_w[i] < 0.05

    def generate_summary_report(self):
        """Generate concise summary report of aggregated comparisons."""
        report = []
        report.append("=" * 100)
        report.append("AGGREGATED STATISTICAL ANALYSIS REPORT")
        report.append("=" * 100)
        report.append("")

        total_comparisons = len(self.results)
        significant_t = sum(1 for combo in self.results.values()
                            if combo['ALL'][0].get('significant_t_test', False))
        significant_w = sum(1 for combo in self.results.values()
                            if combo['ALL'][0].get('significant_wilcoxon', False))

        report.append(f"Total aggregated comparisons: {total_comparisons}")
        report.append(f"Significant (paired t-test): {significant_t}/{total_comparisons}")
        report.append(f"Significant (Wilcoxon): {significant_w}/{total_comparisons}")
        report.append("")

        for combo_name, benchmark_results in self.results.items():
            comp = benchmark_results.get('ALL', [None])[0]
            if comp is None:
                continue
            report.append(f"COMPARISON: {combo_name}")
            report.append(f"  N samples: {comp['n_samples']}")
            report.append(f"  Paired t-test p-value: {comp['t_p_value']:.6e}")
            if 'bonferroni_p_value' in comp:
                report.append(f"    Bonferroni-adjusted: {comp['bonferroni_p_value']:.6e}")
                report.append(f"    FDR-adjusted: {comp['fdr_p_value']:.6e}")
            report.append(f"  Wilcoxon p-value: {comp.get('wilcoxon_p_value', np.nan):.6e}")
            if 'bonferroni_p_value_wilcoxon' in comp:
                report.append(f"    Wilcoxon Bonferroni: {comp['bonferroni_p_value_wilcoxon']:.6e}")
                report.append(f"    Wilcoxon FDR: {comp['fdr_p_value_wilcoxon']:.6e}")
            report.append(f"  Effect size (Cohen's d): {comp['cohens_d']:.4f}")
            report.append("")
        return "\n".join(report)

    def create_visualizations(self, output_dir):
        """Create visualizations of aggregated statistical results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Prepare data
        plot_data = []
        for combo_name, benchmark_results in self.results.items():
            comp = benchmark_results.get('ALL', [None])[0]
            if comp is None:
                continue
            plot_data.append({
                'Comparison': combo_name,
                'Speed Ratio': comp['speed_ratio'],
                'T_P_Value': comp['t_p_value'],
                'Wilcoxon_P_Value': comp.get('wilcoxon_p_value', np.nan),
                'Effect Size': comp['cohens_d'],
                'Significant_T': comp.get('significant_t_test', False),
                'Significant_W': comp.get('significant_wilcoxon', False)
            })

        if not plot_data:
            print("No data available for visualization")
            return

        df = pd.DataFrame(plot_data)

        # P-value bar plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df_sorted = df.sort_values('T_P_Value')
        axes[0].barh(df_sorted['Comparison'], -np.log10(df_sorted['T_P_Value'] + 1e-300))
        axes[0].set_xlabel('-log10 Paired t-test p-value')
        axes[0].set_title('t-test significance')

        df_sorted_w = df.sort_values('Wilcoxon_P_Value')
        axes[1].barh(df_sorted_w['Comparison'], -np.log10(df_sorted_w['Wilcoxon_P_Value'] + 1e-300))
        axes[1].set_xlabel('-log10 Wilcoxon p-value')
        axes[1].set_title('Wilcoxon significance')

        plt.tight_layout()
        plt.savefig(output_dir / 'aggregated_pvalue_significance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualizations saved to {output_dir / 'aggregated_pvalue_significance.png'}")

    def save_detailed_results(self, output_dir):
        """Save aggregated results to CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        rows = []
        for combo_name, benchmark_results in self.results.items():
            comp = benchmark_results.get('ALL', [None])[0]
            if comp is None:
                continue
            row = {
                'Comparison': combo_name,
                'N_Samples': comp['n_samples'],
                'Speed_Ratio': comp['speed_ratio'],
                'T_Statistic': comp['t_statistic'],
                'T_P_Value': comp['t_p_value'],
                'Wilcoxon_P_Value': comp.get('wilcoxon_p_value', np.nan),
                "Cohens_d": comp['cohens_d'],
                'CI_Lower': comp['ci_lower'],
                'CI_Upper': comp['ci_upper'],
                'Significant_T': comp.get('significant_t_test', False),
                'Significant_Wilcoxon': comp.get('significant_wilcoxon', False)
            }
            if 'bonferroni_p_value' in comp:
                row.update({
                    'Bonferroni_T': comp['bonferroni_p_value'],
                    'FDR_T': comp['fdr_p_value'],
                    'Bonferroni_Wilcoxon': comp.get('bonferroni_p_value_wilcoxon', np.nan),
                    'FDR_Wilcoxon': comp.get('fdr_p_value_wilcoxon', np.nan)
                })
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_dir / 'aggregated_statistical_results.csv', index=False)
            print(f"Aggregated results saved to {output_dir / 'aggregated_statistical_results.csv'}")
        else:
            print("No data to save for CSV")

def main():
    # Generate timestamp for default output directory
    default_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    default_output_dir = f'log/{default_timestamp}_statistical_analysis_results'
    
    parser = argparse.ArgumentParser(
        description='Perform aggregated statistical analysis on EAGLE speed comparisons'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        required=True,
        help='Path to tokenizer model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=default_output_dir,
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='+',
        default=['mt_bench', 'humaneval', 'gsm8k', 'alpaca', 'sum', 'qa'],
        help='Benchmarks to aggregate'
    )

    args = parser.parse_args()

    try:
        analyzer = ComprehensiveStatisticalAnalyzer(args.tokenizer_path)

        print(f"Aggregating comparisons across benchmarks in {args.results_dir}...")
        analyzer.aggregate_comparisons_across_benchmarks(args.results_dir, args.benchmarks)

        print("Applying multiple comparison corrections...")
        analyzer.apply_multiple_comparison_corrections()

        print("Generating summary report...")
        report = analyzer.generate_summary_report()

        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / 'aggregated_statistical_analysis_report.txt', 'w') as f:
            f.write(report)

        print("Creating visualizations...")
        analyzer.create_visualizations(output_dir)

        print("Saving detailed results...")
        analyzer.save_detailed_results(output_dir)

        print(f"\nAnalysis complete! Results saved to {output_dir}")
        print("\nSummary report:")
        print(report)

    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
