#!/bin/bash

# EAGLE Overhead vs Performance Trade-off Analysis
# Tests different action cache step sizes to analyze the trade-off between
# RL policy call frequency and inference performance

echo "=== EAGLE Overhead vs Performance Trade-off Analysis ==="
echo "This script tests different action cache step sizes to analyze trade-offs"
echo ""

# Configuration
DATE=$(date '+%Y%m%d_%H%M%S')
TEST_DIR="log/${DATE}_tradeoff_analysis"
mkdir -p "$TEST_DIR"

MODEL_PATH="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
QUESTION_END=80  # Small number for quick analysis
NET_ARCH="128,128"

# Test different cache step sizes
declare -a CACHE_STEPS=(1 5 10 20 50)
declare -a CACHE_DESCRIPTIONS=(
    "No Caching (Every Step)"
    "Light Caching (5 Steps)"
    "Medium Caching (10 Steps)"
    "Heavy Caching (20 Steps)"
    "Max Caching (50 Steps)"
)

echo "Test Directory: $TEST_DIR"
echo "Questions: 0-$QUESTION_END"
echo "Network Architecture: $NET_ARCH"
echo "Cache Step Configurations: ${CACHE_STEPS[*]}"
echo ""

# Create results summary file
RESULTS_FILE="$TEST_DIR/tradeoff_results.csv"
echo "Cache_Steps,Description,LLM_Inference_Time_s,Tokens_Per_Second,RL_Policy_Calls,RL_Policy_Time_ms,RL_Policy_Avg_ms,SentenceBERT_Calls,SentenceBERT_Time_ms,SentenceBERT_Avg_ms,Total_Questions,Total_Overhead_ms" > "$RESULTS_FILE"

# Function to extract metrics from log file
extract_metrics() {
    local log_file=$1
    local cache_steps=$2
    local description="$3"
    
    echo "Extracting metrics from: $log_file"
    
    # Extract overhead analysis section
    if grep -q "Overhead Analysis" "$log_file"; then
        # Get the overhead analysis table
        overhead_section=$(sed -n '/Overhead Analysis/,/TOTAL MEASURED/p' "$log_file")
        
        # Extract LLM inference time (Model_Generation components)
        llm_time=$(echo "$overhead_section" | grep "Model_Generation" | awk '{sum+=$3} END {print sum+0}')
        
        # Extract RL Policy prediction metrics
        rl_calls=$(echo "$overhead_section" | grep "RL_Policy_Prediction" | awk '{sum+=$2} END {print sum+0}')
        rl_time_ms=$(echo "$overhead_section" | grep "RL_Policy_Prediction" | awk '{sum+=$3*1000} END {print sum+0}')
        rl_avg_ms=$(echo "$overhead_section" | grep "RL_Policy_Prediction" | awk '{sum+=$4} END {if(NR>0) print sum/NR; else print 0}')
        
        # Extract SentenceBERT metrics  
        sbert_calls=$(echo "$overhead_section" | grep "SentenceBERT" | awk '{sum+=$2} END {print sum+0}')
        sbert_time_ms=$(echo "$overhead_section" | grep "SentenceBERT" | awk '{sum+=$3*1000} END {print sum+0}')
        sbert_avg_ms=$(echo "$overhead_section" | grep "SentenceBERT" | awk '{sum+=$4} END {if(NR>0) print sum/NR; else print 0}')
        
        # Calculate total overhead
        total_overhead_ms=$(echo "$rl_time_ms + $sbert_time_ms" | bc -l)
        
        # Extract performance metrics
        questions_processed=$(grep -o "Processing questions:.*100%" "$log_file" | tail -1 | grep -o "/[0-9]*" | cut -d'/' -f2)
        if [ -z "$questions_processed" ]; then
            questions_processed=$QUESTION_END
        fi
        
        # Calculate tokens per second (approximate from generation time)
        if [ $(echo "$llm_time > 0" | bc) -eq 1 ]; then
            # Estimate total tokens (approximate)
            total_tokens_estimate=$(echo "$questions_processed * 300" | bc)  # ~300 tokens per question average
            tokens_per_second=$(echo "scale=2; $total_tokens_estimate / $llm_time" | bc)
        else
            tokens_per_second=0
        fi
        
        # Write to CSV
        echo "$cache_steps,$description,$llm_time,$tokens_per_second,$rl_calls,$rl_time_ms,$rl_avg_ms,$sbert_calls,$sbert_time_ms,$sbert_avg_ms,$questions_processed,$total_overhead_ms" >> "$RESULTS_FILE"
        
        echo "  Cache Steps: $cache_steps"
        echo "  LLM Time: ${llm_time}s"
        echo "  Tokens/sec: $tokens_per_second"
        echo "  RL Calls: $rl_calls (${rl_time_ms}ms total, ${rl_avg_ms}ms avg)"
        echo "  SBERT Calls: $sbert_calls (${sbert_time_ms}ms total, ${sbert_avg_ms}ms avg)"
        echo "  Total Overhead: ${total_overhead_ms}ms"
        
    else
        echo "  ❌ No overhead analysis found in log file"
        # Write empty row
        echo "$cache_steps,$description,0,0,0,0,0,0,0,0,$QUESTION_END,0" >> "$RESULTS_FILE"
    fi
    echo ""
}

# Run tests for each cache step configuration
for i in "${!CACHE_STEPS[@]}"; do
    cache_steps="${CACHE_STEPS[$i]}"
    description="${CACHE_DESCRIPTIONS[$i]}"
    test_name="cache_${cache_steps}_steps"
    
    echo "=== Test $((i+1))/${#CACHE_STEPS[@]}: $description (Cache Steps: $cache_steps) ==="
    
    # Create test-specific directory
    test_dir="$TEST_DIR/$test_name"
    mkdir -p "$test_dir/checkpoints"
    
    # Run the test
    PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl_measure \
        --ea-model-path "$MODEL_PATH" \
        --base-model-path "$BASE_MODEL_PATH" \
        --model-id "tradeoff_cache_${cache_steps}" \
        --question-file eagle/data/rl_training/question.jsonl \
        --question-begin 0 \
        --question-end "$QUESTION_END" \
        --answer-file "$test_dir/answers.jsonl" \
        --num-choices 1 \
        --num-gpus-per-model 1 \
        --num-gpus-total 1 \
        --max-gpu-memory "80GiB" \
        --dtype float16 \
        --temperature 0.0 \
        --use-online-rl \
        --use-optimized-sb3-discrete-ppo \
        --optimized-policy-version ofl \
        --online-lr 3e-4 \
        --ppo-net-arch "$NET_ARCH" \
        --enable-max-entropy \
        --max-entropy-ent-coef 0.1 \
        --inference-temperature 1.5 \
        --max-entropy-inference \
        --use-eagle3-features \
        --enable-overhead-measurement \
        --action-cache-steps "$cache_steps" \
        --action-cache-enabled \
        --checkpoint-dir "$test_dir/checkpoints" \
        --no-resume \
        --total-token 60 \
        --depth 7 \
        --top-k 10 \
        --use-stepwise-rl \
        --use-eagle3 2>&1 | tee "$test_dir/overhead_log.txt"
    
    # Extract metrics from this test
    extract_metrics "$test_dir/overhead_log.txt" "$cache_steps" "$description"
    
    echo "Test $((i+1)) completed: $description"
    echo "Waiting 5 seconds before next test..."
    sleep 5
done

echo ""
echo "=== All Trade-off Tests Complete ==="
echo "Results saved to: $RESULTS_FILE"
echo ""

# Generate Python script for visualization
PLOT_SCRIPT="$TEST_DIR/generate_tradeoff_chart.py"
cat > "$PLOT_SCRIPT" << 'EOF'
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def generate_tradeoff_chart(csv_file):
    output_dir = os.path.dirname(csv_file)

    # Load results data
    df = pd.read_csv(csv_file)

    x = np.arange(len(df))
    bar_width = 0.35

    color1 = 'tab:blue'
    color2 = 'tab:green'

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bars1 = ax1.bar(
        x - bar_width/2,
        df['LLM_Inference_Time_s'],
        width=bar_width,
        color=color1,
        alpha=0.7,
        label='Inference Time (s)'
    )

    # Bold, larger font for axis labels
    label_font = {'fontweight': 'bold', 'fontsize': 18}
    tick_font = {'fontweight': 'bold', 'fontsize': 14}

    ax1.set_xlabel('Cache Interval Length', **label_font)
    ax1.set_ylabel('Inference Time (s)', color=color1, **label_font)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=14, width=2)
    ax1.tick_params(axis='x', labelsize=14, width=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Cache_Steps'], fontweight='bold', fontsize=14)
    # Auto-scale y-axis with some padding
    y_min = df['LLM_Inference_Time_s'].min() * 0.95
    y_max = df['LLM_Inference_Time_s'].max() * 1.05
    ax1.set_ylim(y_min, y_max)

    # Make x and y axis lines bold
    ax1.spines['left'].set_linewidth(2.5)
    ax1.spines['bottom'].set_linewidth(2.5)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + bar_width/2,
        df['Tokens_Per_Second'],
        width=bar_width,
        color=color2,
        alpha=0.6,
        label='Generation Speed\n(Tokens/s)'
    )
    ax2.set_ylabel('Generation Speed (Tokens/s)', color=color2, **label_font)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=14, width=2)
    ax2.spines['right'].set_linewidth(2.5)
    # Auto-scale second y-axis with some padding
    y2_min = df['Tokens_Per_Second'].min() * 0.95
    y2_max = df['Tokens_Per_Second'].max() * 1.05
    ax2.set_ylim(y2_min, y2_max)

    # Title and caption
    fig.suptitle(
        'Cache Interval Length vs Inference Time (s) and Generation Speed (Tokens/s)',
        fontsize=16, fontweight='bold', y=0.97
    )

    handles = [bars1, bars2]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='upper left', fontsize=14)

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    output_file = os.path.join(output_dir, 'tradeoff_dualbar.pdf')
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✅ Chart saved to: {output_file}")
    # plt.show()
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_tradeoff_chart.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        sys.exit(1)
    
    print(f"📊 Loading data from: {csv_file}")
    try:
        generate_tradeoff_chart(csv_file)
        print("🎉 Chart generation completed successfully!")
    except Exception as e:
        print(f"❌ Error generating chart: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
EOF

echo "=== Generating Trade-off Visualization ==="
echo "Running Python script to generate chart..."
python3 "$PLOT_SCRIPT" "$RESULTS_FILE"

echo ""
echo "=== Trade-off Analysis Complete ==="
echo ""
echo "📁 Results directory: $TEST_DIR"
echo "📊 Data file: $RESULTS_FILE"
echo "📈 Chart: $TEST_DIR/tradeoff_dualbar.pdf"
echo "🐍 Plot script: $PLOT_SCRIPT"
echo ""
echo "To regenerate the chart:"
echo "  python3 $PLOT_SCRIPT $RESULTS_FILE"
echo ""
echo "Key insights:"
echo "  - Lower cache steps = More RL policy calls = Higher overhead"
echo "  - Higher cache steps = Fewer RL policy calls = Lower overhead"
echo "  - Optimal balance depends on the specific use case"
