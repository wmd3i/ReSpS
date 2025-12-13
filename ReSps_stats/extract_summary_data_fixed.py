#!/usr/bin/env python3
"""
Extract and analyze performance data from summary files
"""

import os
import re
import json
from pathlib import Path

def parse_summary_file(file_path):
    """Parse a summary.txt file and extract performance metrics"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract basic info
    training_date = re.search(r'Training Date: (.+)', content)
    model = re.search(r'Model: (.+)', content)
    base_model = re.search(r'Base Model: (.+)', content)
    execution_mode = re.search(r'Execution Mode: (.+)', content)
    
    # Get policy name from file path (should be the immediate parent directory)
    policy_name = file_path.parent.name
    training_session = file_path.parent.parent.name
    
    # Create a unique identifier that includes the training session
    unique_policy_name = f"{policy_name}_{training_session}"
    
    # If execution_mode is not found in file, try to infer from policy name
    if not execution_mode and policy_name:
        # Create a readable execution mode from policy name
        parts = policy_name.replace('optimized_', '').split('_')
        mode_parts = []
        
        if 'standard' in parts and 'ppo' in parts:
            if 'context' in parts:
                mode_parts.append('Context-Only')
            else:
                mode_parts.append('Standard')
                
            if 'max' in parts and 'entropy' in parts:
                mode_parts.append('Max-Entropy')
            else:
                mode_parts.append('Standard')
                
            if 'ofl' in parts:
                mode_parts.append('OFL Version')
            else:
                mode_parts.append('Standard Version')
                
        execution_mode_text = ' + '.join(mode_parts) if mode_parts else policy_name
    else:
        execution_mode_text = execution_mode.group(1) if execution_mode else policy_name
    
    results = {
        'policy_name': policy_name,
        'unique_policy_name': unique_policy_name,
        'training_session': training_session,
        'file_path': str(file_path),
        'training_date': training_date.group(1) if training_date else None,
        'model': model.group(1) if model else None,
        'base_model': base_model.group(1) if base_model else None,
        'execution_mode': execution_mode_text,
        'benchmarks': {}
    }
    
    # Extract benchmark data
    benchmark_sections = re.split(r'Benchmark: (.+?) - (.+)', content)[1:]
    
    for i in range(0, len(benchmark_sections), 3):
        if i + 2 < len(benchmark_sections):
            benchmark_display = benchmark_sections[i].strip()
            benchmark_id = benchmark_sections[i + 1].strip()
            benchmark_content = benchmark_sections[i + 2]
            
            # Extract speed ratios
            policy_vs_eagle3 = re.search(r'1\. .+ vs EAGLE3 Baseline:.*?Speed ratio \(EAGLE/Baseline\): ([\d.]+)', benchmark_content, re.DOTALL)
            policy_vs_standard = re.search(r'2\. .+ vs Standard LLaMA Baseline:.*?Speed ratio \(EAGLE/Baseline\): ([\d.]+)', benchmark_content, re.DOTALL)
            eagle3_vs_standard = re.search(r'EAGLE3 Baseline vs Standard LLaMA Baseline:.*?Speed ratio \(EAGLE/Baseline\): ([\d.]+)', benchmark_content, re.DOTALL)
            
            # Extract speeds
            policy_speed = re.search(r'1\. .+ vs EAGLE3 Baseline:.*?EAGLE average speed: ([\d.]+)', benchmark_content, re.DOTALL)
            eagle3_speed = re.search(r'1\. .+ vs EAGLE3 Baseline:.*?Baseline average speed: ([\d.]+)', benchmark_content, re.DOTALL)
            standard_speed = re.search(r'2\. .+ vs Standard LLaMA Baseline:.*?Baseline average speed: ([\d.]+)', benchmark_content, re.DOTALL)
            
            results['benchmarks'][benchmark_id] = {
                'display_name': benchmark_display,
                'policy_speed': float(policy_speed.group(1)) if policy_speed else None,
                'eagle3_speed': float(eagle3_speed.group(1)) if eagle3_speed else None,
                'standard_speed': float(standard_speed.group(1)) if standard_speed else None,
                'policy_vs_eagle3_ratio': float(policy_vs_eagle3.group(1)) if policy_vs_eagle3 else None,
                'policy_vs_standard_ratio': float(policy_vs_standard.group(1)) if policy_vs_standard else None,
                'eagle3_vs_standard_ratio': float(eagle3_vs_standard.group(1)) if eagle3_vs_standard else None
            }
    
    return results

def find_summary_files(root_dir):
    """Find all summary.txt files in the directory structure"""
    summary_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'summary.txt':
                summary_files.append(Path(root) / file)
    return summary_files

def create_comparison_table(all_results):
    """Create comparison tables for analysis"""
    
    # Get all unique benchmarks
    all_benchmarks = set()
    for result in all_results:
        all_benchmarks.update(result['benchmarks'].keys())
    
    # Create comparison data
    comparison_data = []
    
    for benchmark in sorted(all_benchmarks):
        benchmark_data = {'benchmark': benchmark}
        
        for result in all_results:
            policy_name = result['unique_policy_name']
            if benchmark in result['benchmarks']:
                bench_data = result['benchmarks'][benchmark]
                benchmark_data[f'{policy_name}_speed'] = bench_data['policy_speed']
                benchmark_data[f'{policy_name}_vs_eagle3'] = bench_data['policy_vs_eagle3_ratio']
                benchmark_data[f'{policy_name}_vs_standard'] = bench_data['policy_vs_standard_ratio']
        
        comparison_data.append(benchmark_data)
    
    return comparison_data

def print_analysis(all_results):
    """Print comprehensive analysis organized by policy/method"""
    
    print("=" * 80)
    print("PERFORMANCE ANALYSIS FROM SUMMARY FILES")
    print("=" * 80)
    print()
    
    # Get all benchmarks for reference
    all_benchmarks = set()
    for result in all_results:
        all_benchmarks.update(result['benchmarks'].keys())
    
    sorted_benchmarks = sorted(all_benchmarks)
    
    print("BENCHMARKS EVALUATED:")
    print("-" * 30)
    for result in all_results:
        for benchmark in sorted_benchmarks:
            if benchmark in result['benchmarks']:
                display_name = result['benchmarks'][benchmark]['display_name']
                print(f"• {benchmark} ({display_name})")
                break
    print()
    
    # Group results by unique training sessions
    session_groups = {}
    for result in all_results:
        session = result['training_session']
        if session not in session_groups:
            session_groups[session] = []
        session_groups[session].append(result)
    
    # Analyze each training session
    for session, session_results in session_groups.items():
        print("=" * 80)
        print(f"TRAINING SESSION: {session.upper()}")
        print("=" * 80)
        print()
        
        # Analyze each policy/method in this session
        for result in session_results:
            print("-" * 80)
            print(f"POLICY: {result['policy_name'].upper()}")
            print("-" * 80)
            print(f"File Path: {result['file_path']}")
            print(f"Execution Mode: {result['execution_mode']}")
            print(f"Model: {result['model']}")
            print(f"Base Model: {result['base_model']}")
            print()
            
            # Performance summary table
            print("PERFORMANCE SUMMARY:")
            print("-" * 70)
            print(f"{'Benchmark':<15} {'vs EAGLE3':<12} {'vs Standard':<12} {'Speed':<12} {'Status'}")
            print("-" * 70)
            
            total_speed = 0
            total_eagle3_ratio = 0
            total_standard_ratio = 0
            count = 0
            
            for benchmark in sorted_benchmarks:
                if benchmark in result['benchmarks']:
                    bench_data = result['benchmarks'][benchmark]
                    speed = bench_data['policy_speed']
                    eagle3_ratio = bench_data['policy_vs_eagle3_ratio']
                    standard_ratio = bench_data['policy_vs_standard_ratio']
                    
                    if speed and eagle3_ratio and standard_ratio:
                        status = "🔥 FASTER" if eagle3_ratio > 1.0 else "🐌 SLOWER"
                        
                        print(f"{benchmark:<15} {eagle3_ratio:<12.4f} {standard_ratio:<12.4f} {speed:<12.1f} {status}")
                        
                        total_speed += speed
                        total_eagle3_ratio += eagle3_ratio
                        total_standard_ratio += standard_ratio
                        count += 1
            
            if count > 0:
                avg_speed = total_speed / count
                avg_eagle3 = total_eagle3_ratio / count
                avg_standard = total_standard_ratio / count
                
                print("-" * 70)
                print(f"{'AVERAGE':<15} {avg_eagle3:<12.4f} {avg_standard:<12.4f} {avg_speed:<12.1f}")
                print()
                
                # Performance insights
                print("PERFORMANCE INSIGHTS:")
                print("-" * 30)
                
                faster_than_eagle3 = sum(1 for b in result['benchmarks'].values() 
                                       if b['policy_vs_eagle3_ratio'] and b['policy_vs_eagle3_ratio'] > 1.0)
                total_benchmarks = len([b for b in result['benchmarks'].values() 
                                      if b['policy_vs_eagle3_ratio']])
                
                print(f"• Faster than EAGLE3 on {faster_than_eagle3}/{total_benchmarks} benchmarks")
                print(f"• Average speedup vs EAGLE3: {avg_eagle3:.4f}x")
                print(f"• Average speedup vs Standard: {avg_standard:.4f}x")
                print(f"• Overall average speed: {avg_speed:.1f} tokens/sec")
                
                # Best and worst performing benchmarks
                best_benchmark = max(result['benchmarks'].items(), 
                                   key=lambda x: x[1]['policy_speed'] if x[1]['policy_speed'] else 0)
                worst_benchmark = min(result['benchmarks'].items(), 
                                    key=lambda x: x[1]['policy_speed'] if x[1]['policy_speed'] else float('inf'))
                
                if best_benchmark[1]['policy_speed'] and worst_benchmark[1]['policy_speed']:
                    print(f"• Best performance: {best_benchmark[0]} ({best_benchmark[1]['policy_speed']:.1f} tok/s)")
                    print(f"• Worst performance: {worst_benchmark[0]} ({worst_benchmark[1]['policy_speed']:.1f} tok/s)")
            
            print()
        
        print()
    
    # Overall ranking and comparison across all sessions
    print("=" * 80)
    print("OVERALL RANKING & COMPARISON (ALL SESSIONS)")
    print("=" * 80)
    
    # Calculate average performance across benchmarks
    policy_averages = {}
    
    for result in all_results:
        unique_name = result['unique_policy_name']
        speeds = []
        vs_eagle3_ratios = []
        vs_standard_ratios = []
        
        for benchmark_data in result['benchmarks'].values():
            if benchmark_data['policy_speed']:
                speeds.append(benchmark_data['policy_speed'])
            if benchmark_data['policy_vs_eagle3_ratio']:
                vs_eagle3_ratios.append(benchmark_data['policy_vs_eagle3_ratio'])
            if benchmark_data['policy_vs_standard_ratio']:
                vs_standard_ratios.append(benchmark_data['policy_vs_standard_ratio'])
        
        if speeds:  # Only include policies with data
            policy_averages[unique_name] = {
                'avg_speed': sum(speeds) / len(speeds),
                'avg_vs_eagle3': sum(vs_eagle3_ratios) / len(vs_eagle3_ratios) if vs_eagle3_ratios else 0,
                'avg_vs_standard': sum(vs_standard_ratios) / len(vs_standard_ratios) if vs_standard_ratios else 0,
                'faster_than_eagle3_count': sum(1 for r in vs_eagle3_ratios if r > 1.0),
                'total_benchmarks': len(vs_eagle3_ratios),
                'session': result['training_session']
            }
    
    # Sort by average speed
    sorted_policies = sorted(policy_averages.items(), key=lambda x: x[1]['avg_speed'], reverse=True)
    
    print("RANKING BY AVERAGE SPEED:")
    print("-" * 40)
    for i, (unique_name, metrics) in enumerate(sorted_policies, 1):
        policy_name = unique_name.split('_')[:-1]  # Remove session suffix
        policy_name = '_'.join(policy_name)
        session = metrics['session']
        faster_ratio = metrics['faster_than_eagle3_count'] / metrics['total_benchmarks'] * 100
        
        print(f"{i}. {policy_name} ({session})")
        print(f"   📊 Avg Speed: {metrics['avg_speed']:.1f} tokens/sec")
        print(f"   ⚡ Avg vs EAGLE3: {metrics['avg_vs_eagle3']:.4f}x")
        print(f"   🚀 Avg vs Standard: {metrics['avg_vs_standard']:.4f}x")
        print(f"   🎯 Faster than EAGLE3: {faster_ratio:.0f}% of benchmarks")
        print()
    
    # Performance comparison matrix
    print("PERFORMANCE COMPARISON MATRIX:")
    print("-" * 50)
    print(f"{'Policy (Session)':<45} {'Avg Speed':<12} {'vs EAGLE3':<12} {'Success Rate'}")
    print("-" * 80)
    
    for unique_name, metrics in sorted_policies:
        policy_name = unique_name.split('_')[:-1]  # Remove session suffix
        policy_name = '_'.join(policy_name)
        session = metrics['session']
        display_name = f"{policy_name} ({session})"
        success_rate = f"{metrics['faster_than_eagle3_count']}/{metrics['total_benchmarks']}"
        print(f"{display_name:<45} {metrics['avg_speed']:<12.1f} {metrics['avg_vs_eagle3']:<12.4f} {success_rate}")
    
    print()
    
    # Find best performing policy for each benchmark
    print("BEST PERFORMER PER BENCHMARK:")
    print("-" * 40)
    
    for benchmark in sorted_benchmarks:
        best_speed = 0
        best_policy = None
        best_session = None
        
        for result in all_results:
            if benchmark in result['benchmarks']:
                speed = result['benchmarks'][benchmark]['policy_speed']
                if speed and speed > best_speed:
                    best_speed = speed
                    best_policy = result['policy_name']
                    best_session = result['training_session']
        
        if best_policy:
            # Get display name
            display_name = None
            for result in all_results:
                if benchmark in result['benchmarks']:
                    display_name = result['benchmarks'][benchmark]['display_name']
                    break
            
            print(f"• {benchmark} ({display_name}): {best_policy} ({best_session}) ({best_speed:.1f} tok/s)")
    
    print()

def main():
    # Find all summary files
    root_dir = "."
    summary_files = find_summary_files(root_dir)
    
    if not summary_files:
        print("No summary.txt files found!")
        return
    
    print(f"Found {len(summary_files)} summary files:")
    for file in summary_files:
        print(f"  - {file}")
    print()
    
    # Parse all files
    all_results = []
    for file_path in summary_files:
        try:
            result = parse_summary_file(file_path)
            all_results.append(result)
            print(f"✓ Parsed: {result['unique_policy_name']}")
        except Exception as e:
            print(f"✗ Error parsing {file_path}: {e}")
    
    print()
    
    if not all_results:
        print("No valid summary files could be parsed!")
        return
    
    # Print analysis
    print_analysis(all_results)
    
    # Save raw data to JSON for further analysis
    output_file = "extracted_performance_data.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw data saved to: {output_file}")

if __name__ == "__main__":
    main()
