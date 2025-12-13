#!/bin/bash

# OPTIMIZED EAGLE DQN Training & Multi-Benchmark Evaluation
# This script demonstrates the optimized DQN policy with:
# 1. EAGLE-3 layer features instead of SBERT text embeddings
# 2. Action caching to reduce computation frequency (every N steps)
# 3. NEW: Context-only state representation option (SBERT embeddings directly)
#  --use-context-only-state \
DATE=$(date '+%Y%m%d_%H%M')
DATE="${DATE}_optimized_dqn"
# DATE='20250725_0725_optimized_dqn'
MODEL_PATH="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
QUESTION_END=1000

# EXECUTION MODE CONFIGURATION
# Set these to "true" or "false" to control which modes to run

# State representation modes
RUN_STANDARD="true"          # Run without --use-context-only-state  
RUN_CONTEXT_ONLY="true"      # Run with --use-context-only-state
RUN_BOTH_STATES="true"      # If true, runs both state modes regardless of above settings

# Entropy modes  
RUN_MAX_ENTROPY="true"       # Run with max-entropy DQN
RUN_NO_MAX_ENTROPY="true"    # Run without max-entropy (standard DQN)
RUN_BOTH_ENTROPY="true"     # If true, runs both entropy modes regardless of above settings

# Benchmark names for testing
BENCHMARKS=("mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa")
BENCHMARK_NAMES=("MT-Bench" "HumanEval" "GSM8K" "Alpaca" "CNN/DailyMail" "Natural Questions")
# BENCHMARKS=("gsm8k" "mt_bench")
# BENCHMARK_NAMES=("GSM8K" "MT-Bench")
# BENCHMARKS=("gsm8k")
# BENCHMARKS=("gsm8k")

# If RUN_BOTH_STATES is true, override individual state settings
if [ "$RUN_BOTH_STATES" = "true" ]; then
    RUN_CONTEXT_ONLY="true"
    RUN_STANDARD="true"
fi

# If RUN_BOTH_ENTROPY is true, override individual entropy settings
if [ "$RUN_BOTH_ENTROPY" = "true" ]; then
    RUN_MAX_ENTROPY="true"
    RUN_NO_MAX_ENTROPY="true"
fi

echo "=== EXECUTION MODE CONFIGURATION ===" | tee -a log/execution_config.txt
echo "STATE MODES:" | tee -a log/execution_config.txt
echo "RUN_CONTEXT_ONLY: $RUN_CONTEXT_ONLY" | tee -a log/execution_config.txt
echo "RUN_STANDARD: $RUN_STANDARD" | tee -a log/execution_config.txt
echo "RUN_BOTH_STATES: $RUN_BOTH_STATES" | tee -a log/execution_config.txt
echo "ENTROPY MODES:" | tee -a log/execution_config.txt
echo "RUN_MAX_ENTROPY: $RUN_MAX_ENTROPY" | tee -a log/execution_config.txt
echo "RUN_NO_MAX_ENTROPY: $RUN_NO_MAX_ENTROPY" | tee -a log/execution_config.txt
echo "RUN_BOTH_ENTROPY: $RUN_BOTH_ENTROPY" | tee -a log/execution_config.txt
echo "" | tee -a log/execution_config.txt

# Create log directory - dynamic based on execution mode
DIRECTORIES_TO_CREATE=()

if [ "$RUN_CONTEXT_ONLY" = "true" ]; then
    if [ "$RUN_MAX_ENTROPY" = "true" ]; then
        DIRECTORIES_TO_CREATE+=("optimized_max_entropy_dqn_context")
    fi
    if [ "$RUN_NO_MAX_ENTROPY" = "true" ]; then
        DIRECTORIES_TO_CREATE+=("optimized_standard_dqn_context")
    fi
fi

if [ "$RUN_STANDARD" = "true" ]; then
    if [ "$RUN_MAX_ENTROPY" = "true" ]; then
        DIRECTORIES_TO_CREATE+=("optimized_max_entropy_dqn_standard")
    fi
    if [ "$RUN_NO_MAX_ENTROPY" = "true" ]; then
        DIRECTORIES_TO_CREATE+=("optimized_standard_dqn_standard")
    fi
fi

# Always create baseline_results directory
DIRECTORIES_TO_CREATE+=("baseline_results")

# Create directories
if [ ${#DIRECTORIES_TO_CREATE[@]} -gt 0 ]; then
    # Create each directory individually to avoid brace expansion issues
    for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
        mkdir -p log/$DATE/$dir
    done
    
    # Create subdirectories for each policy directory (excluding baseline_results)
    for dir in "${DIRECTORIES_TO_CREATE[@]}"; do
        if [ "$dir" != "baseline_results" ]; then
            mkdir -p log/$DATE/$dir/{checkpoints,evaluation}
        fi
    done
fi

echo "=== OPTIMIZED EAGLE DQN Training & Multi-Benchmark Evaluation ===" | tee -a log/$DATE/comparison.txt
echo "Model: $MODEL_PATH" | tee -a log/$DATE/comparison.txt
echo "Base Model: $BASE_MODEL_PATH" | tee -a log/$DATE/comparison.txt
echo "Training Dataset: eagle/data/rl_training/question.jsonl" | tee -a log/$DATE/comparison.txt
echo "OPTIMIZATION 1: EAGLE-3 layer features instead of SBERT embeddings" | tee -a log/$DATE/comparison.txt
echo "OPTIMIZATION 2: Action caching - generate action every 10 steps" | tee -a log/$DATE/comparison.txt
if [ "$RUN_CONTEXT_ONLY" = "true" ]; then
    echo "OPTIMIZATION 3: Context-only state representation (SBERT 384D directly)" | tee -a log/$DATE/comparison.txt
fi
echo "Expected speedup: ~50% reduction in RL policy computation" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# PHASE 1: STANDARD TRAINING (if enabled)
if [ "$RUN_STANDARD" = "true" ]; then
    # Phase 2a: Max-Entropy DQN (Standard) - if enabled
    if [ "$RUN_MAX_ENTROPY" = "true" ]; then
        echo "" | tee -a log/$DATE/comparison.txt
        echo "=== Phase 2a: Training with OPTIMIZED MAX-ENTROPY DQN (Standard) ===" | tee -a log/$DATE/comparison.txt
        echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
        echo "- Action caching every 10 steps" | tee -a log/$DATE/comparison.txt
        echo "- High entropy and temperature-based sampling" | tee -a log/$DATE/comparison.txt
        echo "- Temperature-based inference T=1.5" | tee -a log/$DATE/comparison.txt
        echo "- Full feature state representation (EAGLE-3 + context)" | tee -a log/$DATE/comparison.txt
        echo "- Enhanced exploration during training and inference" | tee -a log/$DATE/comparison.txt
        echo "- DQN parameters: lr=0.01, ε=0.9→0.3, memory=100, batch=8" | tee -a log/$DATE/comparison.txt
        echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
        echo "" | tee -a log/$DATE/comparison.txt

        PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
            --ea-model-path $MODEL_PATH \
            --base-model-path $BASE_MODEL_PATH \
            --model-id optimized_max_entropy_dqn_standard \
            --question-file eagle/data/rl_training/question.jsonl \
            --question-begin 0 \
            --question-end $QUESTION_END \
            --answer-file log/$DATE/optimized_max_entropy_dqn_standard/training_answers.jsonl \
            --num-choices 1 \
            --num-gpus-per-model 1 \
            --num-gpus-total 1 \
            --max-gpu-memory "80GiB" \
            --dtype float16 \
            --temperature 0.0 \
            --use-online-rl \
            --use-optimized-dqn \
            --online-lr 0.01 \
            --online-epsilon-start 0.9 \
            --online-epsilon-end 0.3 \
            --online-memory-size 100 \
            --online-batch-size 8 \
            --action-cache-steps 10 \
            --action-cache-enabled \
            --use-eagle3-features \
            --hidden-size 4096 \
            --checkpoint-dir log/$DATE/optimized_max_entropy_dqn_standard/checkpoints \
            --online-policy-save-path log/$DATE/optimized_max_entropy_dqn_standard/optimized_max_entropy_dqn_policy.pth \
            --checkpoint-freq 500 \
            --wandb-project eagle-optimized-dqn \
            --total-token 60 \
            --depth 7 \
            --top-k 10 \
            --use-stepwise-rl \
            --use-eagle3 2>&1 | tee log/$DATE/optimized_max_entropy_dqn_standard/training.log
    fi

    # Phase 2b: Standard DQN (Standard) - if enabled
    if [ "$RUN_NO_MAX_ENTROPY" = "true" ]; then
        echo "" | tee -a log/$DATE/comparison.txt
        echo "=== Phase 2b: Training with OPTIMIZED STANDARD DQN (Standard) ===" | tee -a log/$DATE/comparison.txt
        echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
        echo "- Action caching every 10 steps" | tee -a log/$DATE/comparison.txt
        echo "- Low temperature and deterministic inference" | tee -a log/$DATE/comparison.txt
        echo "- Standard epsilon-greedy exploration" | tee -a log/$DATE/comparison.txt
        echo "- Full feature state representation (EAGLE-3 + context)" | tee -a log/$DATE/comparison.txt
        echo "- DQN parameters: lr=0.01, ε=0.9→0.3, memory=100, batch=8" | tee -a log/$DATE/comparison.txt
        echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
        echo "" | tee -a log/$DATE/comparison.txt

        PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
            --ea-model-path $MODEL_PATH \
            --base-model-path $BASE_MODEL_PATH \
            --model-id optimized_standard_dqn_standard \
            --question-file eagle/data/rl_training/question.jsonl \
            --question-begin 0 \
            --question-end $QUESTION_END \
            --answer-file log/$DATE/optimized_standard_dqn_standard/training_answers.jsonl \
            --num-choices 1 \
            --num-gpus-per-model 1 \
            --num-gpus-total 1 \
            --max-gpu-memory "80GiB" \
            --dtype float16 \
            --temperature 0.0 \
            --use-online-rl \
            --use-optimized-dqn \
            --online-lr 0.01 \
            --online-epsilon-start 0.9 \
            --online-epsilon-end 0.3 \
            --online-memory-size 100 \
            --online-batch-size 8 \
            --disable-max-entropy \
            --action-cache-steps 10 \
            --action-cache-enabled \
            --use-eagle3-features \
            --hidden-size 4096 \
            --checkpoint-dir log/$DATE/optimized_standard_dqn_standard/checkpoints \
            --online-policy-save-path log/$DATE/optimized_standard_dqn_standard/optimized_standard_dqn_policy.pth \
            --checkpoint-freq 500 \
            --wandb-project eagle-optimized-dqn \
            --total-token 60 \
            --depth 7 \
            --top-k 10 \
            --use-stepwise-rl \
            --use-eagle3 2>&1 | tee log/$DATE/optimized_standard_dqn_standard/training.log
    fi
fi

# PHASE 2: CONTEXT-ONLY TRAINING (if enabled)
if [ "$RUN_CONTEXT_ONLY" = "true" ]; then
    # Phase 1a: Max-Entropy DQN (Context-Only) - if enabled
    if [ "$RUN_MAX_ENTROPY" = "true" ]; then
        echo "=== Phase 1a: Training with OPTIMIZED MAX-ENTROPY DQN (Context-Only) ===" | tee -a log/$DATE/comparison.txt
        echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
        echo "- Action caching every 10 steps" | tee -a log/$DATE/comparison.txt
        echo "- High entropy and temperature-based sampling" | tee -a log/$DATE/comparison.txt
        echo "- Temperature-based inference T=1.5" | tee -a log/$DATE/comparison.txt
        echo "- Context-only state representation (SBERT 384D directly)" | tee -a log/$DATE/comparison.txt
        echo "- Enhanced exploration during training and inference" | tee -a log/$DATE/comparison.txt
        echo "- DQN parameters: lr=0.01, ε=0.9→0.3, memory=100, batch=8" | tee -a log/$DATE/comparison.txt
        echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
        echo "" | tee -a log/$DATE/comparison.txt

        PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
            --ea-model-path $MODEL_PATH \
            --base-model-path $BASE_MODEL_PATH \
            --model-id optimized_max_entropy_dqn_context \
            --question-file eagle/data/rl_training/question.jsonl \
            --question-begin 0 \
            --question-end $QUESTION_END \
            --answer-file log/$DATE/optimized_max_entropy_dqn_context/training_answers.jsonl \
            --num-choices 1 \
            --num-gpus-per-model 1 \
            --num-gpus-total 1 \
            --max-gpu-memory "80GiB" \
            --dtype float16 \
            --temperature 0.0 \
            --use-online-rl \
            --use-optimized-dqn \
            --online-lr 0.01 \
            --online-epsilon-start 0.9 \
            --online-epsilon-end 0.3 \
            --online-memory-size 100 \
            --online-batch-size 8 \
            --action-cache-steps 10 \
            --action-cache-enabled \
            --use-eagle3-features \
            --hidden-size 4096 \
            --use-context-only-state \
            --checkpoint-dir log/$DATE/optimized_max_entropy_dqn_context/checkpoints \
            --online-policy-save-path log/$DATE/optimized_max_entropy_dqn_context/optimized_max_entropy_dqn_policy.pth \
            --checkpoint-freq 500 \
            --wandb-project eagle-optimized-dqn \
            --total-token 60 \
            --depth 7 \
            --top-k 10 \
            --use-stepwise-rl \
            --use-eagle3 2>&1 | tee log/$DATE/optimized_max_entropy_dqn_context/training.log
    fi

    # Phase 1b: Standard DQN (Context-Only) - if enabled
    if [ "$RUN_NO_MAX_ENTROPY" = "true" ]; then
        echo "" | tee -a log/$DATE/comparison.txt
        echo "=== Phase 1b: Training with OPTIMIZED STANDARD DQN (Context-Only) ===" | tee -a log/$DATE/comparison.txt
        echo "- EAGLE-3 layer features for state representation" | tee -a log/$DATE/comparison.txt
        echo "- Action caching every 10 steps" | tee -a log/$DATE/comparison.txt
        echo "- Low temperature and deterministic inference" | tee -a log/$DATE/comparison.txt
        echo "- Standard epsilon-greedy exploration" | tee -a log/$DATE/comparison.txt
        echo "- Context-only state representation (SBERT 384D directly)" | tee -a log/$DATE/comparison.txt
        echo "- DQN parameters: lr=0.01, ε=0.9→0.3, memory=100, batch=8" | tee -a log/$DATE/comparison.txt
        echo "- Training dataset: questions 0-$QUESTION_END for faster training" | tee -a log/$DATE/comparison.txt
        echo "" | tee -a log/$DATE/comparison.txt

        PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
            --ea-model-path $MODEL_PATH \
            --base-model-path $BASE_MODEL_PATH \
            --model-id optimized_standard_dqn_context \
            --question-file eagle/data/rl_training/question.jsonl \
            --question-begin 0 \
            --question-end $QUESTION_END \
            --answer-file log/$DATE/optimized_standard_dqn_context/training_answers.jsonl \
            --num-choices 1 \
            --num-gpus-per-model 1 \
            --num-gpus-total 1 \
            --max-gpu-memory "80GiB" \
            --dtype float16 \
            --temperature 0.0 \
            --use-online-rl \
            --use-optimized-dqn \
            --online-lr 0.01 \
            --online-epsilon-start 0.9 \
            --online-epsilon-end 0.3 \
            --online-memory-size 100 \
            --online-batch-size 8 \
            --disable-max-entropy \
            --action-cache-steps 10 \
            --action-cache-enabled \
            --use-eagle3-features \
            --hidden-size 4096 \
            --use-context-only-state \
            --checkpoint-dir log/$DATE/optimized_standard_dqn_context/checkpoints \
            --online-policy-save-path log/$DATE/optimized_standard_dqn_context/optimized_standard_dqn_policy.pth \
            --checkpoint-freq 500 \
            --wandb-project eagle-optimized-dqn \
            --total-token 60 \
            --depth 7 \
            --top-k 10 \
            --use-stepwise-rl \
            --use-eagle3 2>&1 | tee log/$DATE/optimized_standard_dqn_context/training.log
    fi
fi

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 3: Multi-Benchmark Evaluation ===" | tee -a log/$DATE/comparison.txt

# Count total number of models to evaluate
TOTAL_MODELS=0
if [ "$RUN_CONTEXT_ONLY" = "true" ]; then
    if [ "$RUN_MAX_ENTROPY" = "true" ]; then
        TOTAL_MODELS=$((TOTAL_MODELS + 1))  # max-entropy context
    fi
    if [ "$RUN_NO_MAX_ENTROPY" = "true" ]; then
        TOTAL_MODELS=$((TOTAL_MODELS + 1))  # standard context
    fi
fi
if [ "$RUN_STANDARD" = "true" ]; then
    if [ "$RUN_MAX_ENTROPY" = "true" ]; then
        TOTAL_MODELS=$((TOTAL_MODELS + 1))  # max-entropy standard
    fi
    if [ "$RUN_NO_MAX_ENTROPY" = "true" ]; then
        TOTAL_MODELS=$((TOTAL_MODELS + 1))  # standard standard
    fi
fi

echo "Testing $TOTAL_MODELS optimized trained policies on ${#BENCHMARKS[@]} benchmarks:" | tee -a log/$DATE/comparison.txt
for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    benchmark_name="${BENCHMARK_NAMES[$i]}"
    echo "$((i+1)). $benchmark_name - $benchmark" | tee -a log/$DATE/comparison.txt
done
echo "" | tee -a log/$DATE/comparison.txt

# Check if trained policies exist and create evaluation plan
POLICIES_TO_EVALUATE=()
POLICY_LABELS=()

if [ "$RUN_CONTEXT_ONLY" = "true" ]; then
    if [ "$RUN_MAX_ENTROPY" = "true" ] && [ -f "log/$DATE/optimized_max_entropy_dqn_context/optimized_max_entropy_dqn_policy.pth" ]; then
        POLICIES_TO_EVALUATE+=("optimized_max_entropy_dqn_context")
        POLICY_LABELS+=("Max-Entropy DQN (Context-Only)")
    elif [ "$RUN_MAX_ENTROPY" = "true" ]; then
        echo "❌ Context-only max-entropy DQN policy not found!" | tee -a log/$DATE/comparison.txt
    fi
    
    if [ "$RUN_NO_MAX_ENTROPY" = "true" ] && [ -f "log/$DATE/optimized_standard_dqn_context/optimized_standard_dqn_policy.pth" ]; then
        POLICIES_TO_EVALUATE+=("optimized_standard_dqn_context")
        POLICY_LABELS+=("Standard DQN (Context-Only)")
    elif [ "$RUN_NO_MAX_ENTROPY" = "true" ]; then
        echo "❌ Context-only standard DQN policy not found!" | tee -a log/$DATE/comparison.txt
    fi
fi

if [ "$RUN_STANDARD" = "true" ]; then
    if [ "$RUN_MAX_ENTROPY" = "true" ] && [ -f "log/$DATE/optimized_max_entropy_dqn_standard/optimized_max_entropy_dqn_policy.pth" ]; then
        POLICIES_TO_EVALUATE+=("optimized_max_entropy_dqn_standard")
        POLICY_LABELS+=("Max-Entropy DQN (Standard)")
    elif [ "$RUN_MAX_ENTROPY" = "true" ]; then
        echo "❌ Standard max-entropy DQN policy not found!" | tee -a log/$DATE/comparison.txt
    fi
    
    if [ "$RUN_NO_MAX_ENTROPY" = "true" ] && [ -f "log/$DATE/optimized_standard_dqn_standard/optimized_standard_dqn_policy.pth" ]; then
        POLICIES_TO_EVALUATE+=("optimized_standard_dqn_standard")
        POLICY_LABELS+=("Standard DQN (Standard)")
    elif [ "$RUN_NO_MAX_ENTROPY" = "true" ]; then
        echo "❌ Standard DQN policy not found!" | tee -a log/$DATE/comparison.txt
    fi
fi

if [ ${#POLICIES_TO_EVALUATE[@]} -eq 0 ]; then
    echo "❌ No trained policies found for evaluation!" | tee -a log/$DATE/comparison.txt
    exit 1
fi

echo "✅ Found ${#POLICIES_TO_EVALUATE[@]} trained policies. Starting evaluation..." | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Evaluate all policies on all benchmarks
for j in "${!POLICIES_TO_EVALUATE[@]}"; do
    policy_dir="${POLICIES_TO_EVALUATE[$j]}"
    policy_label="${POLICY_LABELS[$j]}"
    
    for i in "${!BENCHMARKS[@]}"; do
        benchmark="${BENCHMARKS[$i]}"
        benchmark_name="${BENCHMARK_NAMES[$i]}"
        
        echo "=== Evaluating $benchmark_name with $policy_label ===" | tee -a log/$DATE/comparison.txt

        if [ ! -f log/$DATE/$policy_dir/evaluation/${benchmark}_results.jsonl ]; then
            # Determine if this policy uses context-only mode
            CONTEXT_ARGS=""
            if [[ "$policy_dir" == *"_context" ]]; then
                CONTEXT_ARGS="--use-context-only-state"
            fi
            
            # Determine if this policy uses max-entropy mode
            ENTROPY_ARGS=""
            if [[ "$policy_dir" == *"max_entropy"* ]]; then
                ENTROPY_ARGS=""  # DQN doesn't use the same entropy args as PPO
            else
                ENTROPY_ARGS="--disable-max-entropy"
            fi
            
            python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
                --ea-model-path $MODEL_PATH \
                --base-model-path $BASE_MODEL_PATH \
                --model-id ${policy_dir}_$benchmark \
                --question-file eagle/data/$benchmark/question.jsonl \
                --question-begin 0 \
                --question-end -1 \
                --answer-file log/$DATE/$policy_dir/evaluation/${benchmark}_results.jsonl \
                --num-choices 1 \
                --num-gpus-per-model 1 \
                --num-gpus-total 1 \
                --max-gpu-memory "80GiB" \
                --dtype float16 \
                --temperature 0.0 \
                --use-online-rl \
                --use-optimized-dqn \
                --online-inference-only \
                --online-policy-path log/$DATE/$policy_dir/optimized_*_dqn_policy.pth \
                $CONTEXT_ARGS \
                $ENTROPY_ARGS \
                --action-cache-steps 30 \
                --action-cache-enabled \
                --use-eagle3-features \
                --hidden-size 4096 \
                --total-token 60 \
                --depth 7 \
                --top-k 10 \
                --use-stepwise-rl \
                --use-eagle3 2>&1 | tee log/$DATE/$policy_dir/evaluation/${benchmark}_evaluation.log
        else
            echo "Results already exist for $policy_label on $benchmark_name" | tee -a log/$DATE/comparison.txt
        fi
    done
done


echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 4: Performance Analysis Across All Benchmarks ===" | tee -a log/$DATE/comparison.txt
echo "Generating baseline results for comprehensive comparison..." | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Create consolidated results summary
echo "Performance Summary Report" >> log/$DATE/summary.txt
echo "=========================" >> log/$DATE/summary.txt
echo "Training Date: $DATE" >> log/$DATE/summary.txt
echo "Model: $MODEL_PATH" >> log/$DATE/summary.txt
echo "Base Model: $BASE_MODEL_PATH" >> log/$DATE/summary.txt
echo "" >> log/$DATE/summary.txt

# Generate baseline results for all benchmarks (LLaMA 3.1 8B)
echo "=== Generating LLaMA 3.1 8B Baseline Results ===" | tee -a log/$DATE/comparison.txt

for benchmark in "${BENCHMARKS[@]}"; do
    echo "Generating baseline for $benchmark..." | tee -a log/$DATE/comparison.txt
    
    # Generate EAGLE3 baseline
    if [ ! -f "log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl" ]; then
        python -m eagle.evaluation.gen_ea_answer_llama3chat \
            --ea-model-path "$MODEL_PATH" \
            --base-model-path "$BASE_MODEL_PATH" \
            --bench-name "$benchmark" \
            --answer-file "log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl" \
            --temperature 0.0 \
            --use_eagle3 \
            2>&1 | tee -a log/$DATE/baseline_results/baseline_${benchmark}_eagle3.log
    else
        echo "EAGLE3 baseline for $benchmark already exists" | tee -a log/$DATE/comparison.txt
    fi
    
    # Generate standard baseline
    if [ ! -f "log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl" ]; then
        python -m eagle.evaluation.gen_baseline_answer_llama3chat \
            --ea-model-path "$MODEL_PATH" \
            --base-model-path "$BASE_MODEL_PATH" \
            --bench-name "$benchmark" \
            --answer-file "log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl" \
            --temperature 0.0 \
            2>&1 | tee -a log/$DATE/baseline_results/baseline_${benchmark}_standard.log
    else
        echo "Standard baseline for $benchmark already exists" | tee -a log/$DATE/comparison.txt
    fi
done

# Analyze results for each benchmark with comprehensive comparisons
echo "" | tee -a log/$DATE/comparison.txt
echo "=== Comprehensive Performance Analysis ===" | tee -a log/$DATE/comparison.txt

for i in "${!BENCHMARKS[@]}"; do
    benchmark="${BENCHMARKS[$i]}"
    benchmark_name="${BENCHMARK_NAMES[$i]}"
    
    echo "=== $benchmark_name - $benchmark Performance Analysis ===" | tee -a log/$DATE/comparison.txt
    echo "Benchmark: $benchmark_name - $benchmark" >> log/$DATE/summary.txt
    echo "===========================================" >> log/$DATE/summary.txt
    
    # Define baseline files
    eagle3_file="log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl"
    baseline_file="log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl"
    
    # Create list of result files for this benchmark
    RESULT_FILES=()
    RESULT_LABELS=()
    
    for j in "${!POLICIES_TO_EVALUATE[@]}"; do
        policy_dir="${POLICIES_TO_EVALUATE[$j]}"
        policy_label="${POLICY_LABELS[$j]}"
        result_file="log/$DATE/$policy_dir/evaluation/${benchmark}_results.jsonl"
        
        if [ -f "$result_file" ]; then
            RESULT_FILES+=("$result_file")
            RESULT_LABELS+=("$policy_label")
        fi
    done
    
    if [ ${#RESULT_FILES[@]} -gt 0 ] && [ -f "$eagle3_file" ] && [ -f "$baseline_file" ]; then
        echo "✅ Found ${#RESULT_FILES[@]} policy results and baseline files for $benchmark_name" | tee -a log/$DATE/comparison.txt
        
        # Speed comparison using existing speed.py tool
        if [ -f "eagle/evaluation/speed.py" ]; then
            echo "" >> log/$DATE/summary.txt
            
            # Compare each policy against baselines
            for k in "${!RESULT_FILES[@]}"; do
                policy_file="${RESULT_FILES[$k]}"
                policy_label="${RESULT_LABELS[$k]}"
                
                echo "$((k*2+1)). $policy_label vs EAGLE3 Baseline:" >> log/$DATE/summary.txt
                python eagle/evaluation/speed.py \
                    --ea-file "$policy_file" \
                    --baseline-file "$eagle3_file" \
                    --tokenizer-path "$BASE_MODEL_PATH" \
                    2>&1 | tee -a log/$DATE/summary.txt
                
                echo "" >> log/$DATE/summary.txt
                echo "$((k*2+2)). $policy_label vs Standard LLaMA Baseline:" >> log/$DATE/summary.txt
                python eagle/evaluation/speed.py \
                    --ea-file "$policy_file" \
                    --baseline-file "$baseline_file" \
                    --tokenizer-path "$BASE_MODEL_PATH" \
                    2>&1 | tee -a log/$DATE/summary.txt
                echo "" >> log/$DATE/summary.txt
            done
            
            # Compare baselines against each other
            echo "EAGLE3 Baseline vs Standard LLaMA Baseline:" >> log/$DATE/summary.txt
            python eagle/evaluation/speed.py \
                --ea-file "$eagle3_file" \
                --baseline-file "$baseline_file" \
                --tokenizer-path "$BASE_MODEL_PATH" \
                2>&1 | tee -a log/$DATE/summary.txt
            
            # Compare policies against each other if we have multiple
            if [ ${#RESULT_FILES[@]} -gt 1 ]; then
                echo "" >> log/$DATE/summary.txt
                echo "Policy Comparisons:" >> log/$DATE/summary.txt
                for k in "${!RESULT_FILES[@]}"; do
                    for l in "${!RESULT_FILES[@]}"; do
                        if [ $k -lt $l ]; then
                            policy1_file="${RESULT_FILES[$k]}"
                            policy1_label="${RESULT_LABELS[$k]}"
                            policy2_file="${RESULT_FILES[$l]}"
                            policy2_label="${RESULT_LABELS[$l]}"
                            
                            echo "$policy1_label vs $policy2_label:" >> log/$DATE/summary.txt
                            python eagle/evaluation/speed.py \
                                --ea-file "$policy1_file" \
                                --baseline-file "$policy2_file" \
                                --tokenizer-path "$BASE_MODEL_PATH" \
                                2>&1 | tee -a log/$DATE/summary.txt
                            echo "" >> log/$DATE/summary.txt
                        fi
                    done
                done
            fi
                
        else
            echo "Speed analysis tool not found" >> log/$DATE/summary.txt
        fi
        
        # Basic statistics
        echo "" >> log/$DATE/summary.txt
        echo "Result File Statistics:" >> log/$DATE/summary.txt
        for k in "${!RESULT_FILES[@]}"; do
            policy_file="${RESULT_FILES[$k]}"
            policy_label="${RESULT_LABELS[$k]}"
            echo "$policy_label: $(wc -l < "$policy_file") samples" >> log/$DATE/summary.txt
        done
        echo "EAGLE3 Baseline: $(wc -l < "$eagle3_file") samples" >> log/$DATE/summary.txt
        echo "Standard Baseline: $(wc -l < "$baseline_file") samples" >> log/$DATE/summary.txt
        
    else
        echo "❌ Missing result files for $benchmark_name" | tee -a log/$DATE/comparison.txt
        echo "Required files:" | tee -a log/$DATE/comparison.txt
        for k in "${!RESULT_FILES[@]}"; do
            policy_file="${RESULT_FILES[$k]}"
            policy_label="${RESULT_LABELS[$k]}"
            echo "  $policy_label: $policy_file $([ -f "$policy_file" ] && echo "✅" || echo "❌")" | tee -a log/$DATE/comparison.txt
        done
        echo "  EAGLE3 Baseline: $eagle3_file $([ -f "$eagle3_file" ] && echo "✅" || echo "❌")" | tee -a log/$DATE/comparison.txt
        echo "  Standard Baseline: $baseline_file $([ -f "$baseline_file" ] && echo "✅" || echo "❌")" | tee -a log/$DATE/comparison.txt
        echo "Missing result files - check evaluation and baseline generation logs" >> log/$DATE/summary.txt
    fi
    
    echo "" >> log/$DATE/summary.txt
    echo "" | tee -a log/$DATE/comparison.txt
done

# Count policies trained and evaluated for summary
TRAINED_POLICIES=0
if [ "$RUN_CONTEXT_ONLY" = "true" ]; then
    if [ "$RUN_MAX_ENTROPY" = "true" ]; then
        ((TRAINED_POLICIES += 1))
    fi
    if [ "$RUN_NO_MAX_ENTROPY" = "true" ]; then
        ((TRAINED_POLICIES += 1))
    fi
fi
if [ "$RUN_STANDARD" = "true" ]; then
    if [ "$RUN_MAX_ENTROPY" = "true" ]; then
        ((TRAINED_POLICIES += 1))
    fi
    if [ "$RUN_NO_MAX_ENTROPY" = "true" ]; then
        ((TRAINED_POLICIES += 1))
    fi
fi

# Determine execution mode description
EXECUTION_MODE=""
if [ "$RUN_CONTEXT_ONLY" = "true" ] && [ "$RUN_STANDARD" = "true" ]; then
    EXECUTION_MODE="Both States"
elif [ "$RUN_CONTEXT_ONLY" = "true" ]; then
    EXECUTION_MODE="Context-Only"
elif [ "$RUN_STANDARD" = "true" ]; then
    EXECUTION_MODE="Standard"
fi

if [ "$RUN_MAX_ENTROPY" = "true" ] && [ "$RUN_NO_MAX_ENTROPY" = "true" ]; then
    EXECUTION_MODE="$EXECUTION_MODE + Both Entropy"
elif [ "$RUN_MAX_ENTROPY" = "true" ]; then
    EXECUTION_MODE="$EXECUTION_MODE + Max-Entropy"
elif [ "$RUN_NO_MAX_ENTROPY" = "true" ]; then
    EXECUTION_MODE="$EXECUTION_MODE + Standard"
fi

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Summary ===" | tee -a log/$DATE/comparison.txt
echo "Training completed with optimized DQN policies!" | tee -a log/$DATE/comparison.txt
echo "Execution Mode: $EXECUTION_MODE" | tee -a log/$DATE/comparison.txt
echo "Policies trained: $TRAINED_POLICIES" | tee -a log/$DATE/comparison.txt
echo "Results saved in: log/$DATE/" | tee -a log/$DATE/comparison.txt
echo "Key optimizations implemented:" | tee -a log/$DATE/comparison.txt
echo "1. EAGLE-3 layer features with optional context-only mode" | tee -a log/$DATE/comparison.txt
echo "2. Action caching every 10 steps (~50% computation reduction)" | tee -a log/$DATE/comparison.txt
echo "3. Flexible execution modes for targeted experiments" | tee -a log/$DATE/comparison.txt
echo "4. Enhanced DQN with temperature-based action sampling" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Create performance summary
echo "Performance Summary Report" >> log/$DATE/summary.txt
echo "=========================" >> log/$DATE/summary.txt
echo "Date: $DATE" >> log/$DATE/summary.txt
echo "Execution Mode: $EXECUTION_MODE" >> log/$DATE/summary.txt
echo "Policies Trained: $TRAINED_POLICIES" >> log/$DATE/summary.txt
echo "Model: $MODEL_PATH" >> log/$DATE/summary.txt
echo "Algorithm: Optimized DQN" >> log/$DATE/summary.txt
echo "Optimizations: EAGLE-3 features + Action caching + Flexible execution" >> log/$DATE/summary.txt
echo "Training questions: $QUESTION_END" >> log/$DATE/summary.txt
echo "Benchmarks evaluated: ${BENCHMARKS[*]}" >> log/$DATE/summary.txt
echo "Policies evaluated: ${#POLICIES_TO_EVALUATE[@]}" >> log/$DATE/summary.txt
for j in "${!POLICY_LABELS[@]}"; do
    echo "  - ${POLICY_LABELS[$j]}" >> log/$DATE/summary.txt
done
echo "" >> log/$DATE/summary.txt

echo "Check log/$DATE/summary.txt for detailed results." | tee -a log/$DATE/comparison.txt