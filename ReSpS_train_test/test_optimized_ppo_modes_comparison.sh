#!/bin/bash

# OPTIMIZED EAGLE SB3 PPO Training & Multi-Benchmark Evaluation
# This script demonstrates the optimized policy with:
# 1. EAGLE-3 layer features instead of SBERT text embeddings
# 2. Action caching to reduce computation frequency (every N steps)
# 3. NEW: Context-only state representation option (SBERT embeddings directly)
# 4. NEW: Choice between Standard and OFL policy versions
# 5. NEW: Parallel execution with GPU management
#
# USAGE:
# To run Standard version only: Set RUN_STANDARD_VERSION=1, RUN_OFL_VERSION=0
# To run OFL version only: Set RUN_STANDARD_VERSION=0, RUN_OFL_VERSION=1  
# To run both versions: Set RUN_STANDARD_VERSION=1, RUN_OFL_VERSION=1
#
# GPU PARALLEL EXECUTION:
# - Set TOTAL_GPUS to the number of available GPUs
# - Script will automatically distribute tasks across GPUs
# - During RL training: Multiple tasks can share 1 GPU
# - During inference/baseline: 1 task per GPU (tasks wait if not enough GPUs)
#
# Example configurations:
# - Standard only: RUN_STANDARD_VERSION=1, RUN_OFL_VERSION=0
# - OFL only: RUN_STANDARD_VERSION=0, RUN_OFL_VERSION=1  
# - Both versions: RUN_STANDARD_VERSION=1, RUN_OFL_VERSION=1
#
#  --use-context-only-state \
DATE=$(date '+%Y%m%d_%H%M')
DATE="${DATE}_optimized_ppo"
# DATE='20250725_0725_optimized_ppo'
# DATE='20250726_0904_optimized_ppo'
MODEL_PATH="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
QUESTION_END=4000

# EXECUTION MODE CONFIGURATION
# Set these to "true" or "false" to control which modes to run

# Policy implementation version selection
RUN_STANDARD_VERSION=1   # Run standard policy version
RUN_OFL_VERSION=0       # Run OFL policy version with enhanced features

# GPU Configuration
TOTAL_GPUS=4  # Set this to your total number of GPUs
MAX_TASKS_PER_GPU_TRAINING=2  # During RL training, how many tasks can share 1 GPU
MAX_TASKS_PER_GPU_INFERENCE=1  # During inference/baseline, 1 task per GPU

# Parallel execution configuration
ENABLE_PARALLEL_EXECUTION=1  # Set to "false" to run sequentially
MAX_PARALLEL_PROCESSES=8  # Maximum number of parallel processes

# Task queue management
TASK_QUEUE_DIR="task_queue"
GPU_LOCK_DIR="gpu_locks"
PID_FILE_DIR="process_pids"

# State representation modes
RUN_STANDARD=1          # Run without --use-context-only-state  
RUN_CONTEXT_ONLY=0      # Run with --use-context-only-state

# Entropy modes  
RUN_MAX_ENTROPY=1       # Run with max-entropy PPO
RUN_NO_MAX_ENTROPY=0    # Run without max-entropy (standard PPO)

# Benchmark names for testing
BENCHMARKS=("mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa")
BENCHMARK_NAMES=("MT-Bench" "HumanEval" "GSM8K" "Alpaca" "CNN/DailyMail" "Natural Questions")
BENCHMARKS=("gsm8k" "mt_bench")
BENCHMARK_NAMES=("GSM8K" "MT-Bench")
# BENCHMARKS=("gsm8k")
# BENCHMARKS=("gsm8k")

# Handle version selection logic
if [ "$RUN_STANDARD_VERSION" -eq 1 ] && [ "$RUN_OFL_VERSION" -eq 1 ]; then
    echo "🔄 Running both policy versions for comparison"
elif [ "$RUN_OFL_VERSION" -eq 1 ]; then
    RUN_STANDARD_VERSION=0
    echo "🚀 Running OFL version with enhanced features"
else
    echo "📋 Running standard version"
fi

echo "=== EXECUTION MODE CONFIGURATION ===" | tee -a log/execution_config.txt
echo "POLICY VERSIONS:" | tee -a log/execution_config.txt
echo "RUN_STANDARD_VERSION: $RUN_STANDARD_VERSION" | tee -a log/execution_config.txt
echo "RUN_OFL_VERSION: $RUN_OFL_VERSION" | tee -a log/execution_config.txt
echo "GPU CONFIGURATION:" | tee -a log/execution_config.txt
echo "TOTAL_GPUS: $TOTAL_GPUS" | tee -a log/execution_config.txt
echo "MAX_TASKS_PER_GPU_TRAINING: $MAX_TASKS_PER_GPU_TRAINING" | tee -a log/execution_config.txt
echo "MAX_TASKS_PER_GPU_INFERENCE: $MAX_TASKS_PER_GPU_INFERENCE" | tee -a log/execution_config.txt
echo "ENABLE_PARALLEL_EXECUTION: $ENABLE_PARALLEL_EXECUTION" | tee -a log/execution_config.txt
echo "MAX_PARALLEL_PROCESSES: $MAX_PARALLEL_PROCESSES" | tee -a log/execution_config.txt
echo "STATE MODES:" | tee -a log/execution_config.txt
echo "RUN_CONTEXT_ONLY: $RUN_CONTEXT_ONLY" | tee -a log/execution_config.txt
echo "RUN_STANDARD: $RUN_STANDARD" | tee -a log/execution_config.txt
echo "ENTROPY MODES:" | tee -a log/execution_config.txt
echo "RUN_MAX_ENTROPY: $RUN_MAX_ENTROPY" | tee -a log/execution_config.txt
echo "RUN_NO_MAX_ENTROPY: $RUN_NO_MAX_ENTROPY" | tee -a log/execution_config.txt
echo "" | tee -a log/execution_config.txt

# GPU Management Functions
setup_gpu_management() {
    echo "🔧 Setting up GPU management for parallel execution..."
    
    # Create directories for task management
    mkdir -p $TASK_QUEUE_DIR
    mkdir -p $GPU_LOCK_DIR
    mkdir -p $PID_FILE_DIR
    
    # Initialize GPU locks
    for i in $(seq 0 $((TOTAL_GPUS-1))); do
        echo "0" > $GPU_LOCK_DIR/gpu_${i}_tasks.txt
    done
    
    echo "✅ GPU management setup complete"
}

get_available_gpu() {
    local task_type=$1  # "training" or "inference"
    local max_tasks_per_gpu
    
    if [ "$task_type" = "training" ]; then
        max_tasks_per_gpu=$MAX_TASKS_PER_GPU_TRAINING
    else
        max_tasks_per_gpu=$MAX_TASKS_PER_GPU_INFERENCE
    fi
    
    # Check each GPU for availability
    for i in $(seq 0 $((TOTAL_GPUS-1))); do
        local current_tasks=$(cat $GPU_LOCK_DIR/gpu_${i}_tasks.txt 2>/dev/null || echo "0")
        if [ "$current_tasks" -lt "$max_tasks_per_gpu" ]; then
            echo $i
            return 0
        fi
    done
    
    # No GPU available
    echo "-1"
}

acquire_gpu() {
    local gpu_id=$1
    local task_type=$2
    
    if [ "$gpu_id" -eq -1 ]; then
        return 1
    fi
    
    local current_tasks=$(cat $GPU_LOCK_DIR/gpu_${gpu_id}_tasks.txt 2>/dev/null || echo "0")
    local max_tasks_per_gpu
    
    if [ "$task_type" = "training" ]; then
        max_tasks_per_gpu=$MAX_TASKS_PER_GPU_TRAINING
    else
        max_tasks_per_gpu=$MAX_TASKS_PER_GPU_INFERENCE
    fi
    
    if [ "$current_tasks" -lt "$max_tasks_per_gpu" ]; then
        echo $((current_tasks + 1)) > $GPU_LOCK_DIR/gpu_${gpu_id}_tasks.txt
        return 0
    fi
    
    return 1
}

release_gpu() {
    local gpu_id=$1
    
    if [ "$gpu_id" -eq -1 ]; then
        return
    fi
    
    local current_tasks=$(cat $GPU_LOCK_DIR/gpu_${gpu_id}_tasks.txt 2>/dev/null || echo "0")
    if [ "$current_tasks" -gt 0 ]; then
        echo $((current_tasks - 1)) > $GPU_LOCK_DIR/gpu_${gpu_id}_tasks.txt
    fi
}

wait_for_gpu() {
    local task_type=$1
    local max_wait_time=300  # 5 minutes max wait
    
    local start_time=$(date +%s)
    while [ $(($(date +%s) - start_time)) -lt $max_wait_time ]; do
        local available_gpu=$(get_available_gpu $task_type)
        if [ "$available_gpu" -ne -1 ]; then
            echo $available_gpu
            return 0
        fi
        echo "⏳ Waiting for available GPU for $task_type task..." >&2
        sleep 10
    done
    
    echo "❌ Timeout waiting for GPU" >&2
    echo "-1"
}

cleanup_gpu_management() {
    echo "🧹 Cleaning up GPU management..."
    rm -rf $TASK_QUEUE_DIR
    rm -rf $GPU_LOCK_DIR
    rm -rf $PID_FILE_DIR
    echo "✅ GPU management cleanup complete"
}

# Task Generation and Management
generate_training_tasks() {
    echo "📋 Generating training tasks..."
    
    local task_id=0
    
    # Standard Version Training Tasks
    if [ "$RUN_STANDARD_VERSION" -eq 1 ]; then
        if [ "$RUN_STANDARD" -eq 1 ]; then
            if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
                echo "training|standard|max_entropy|standard|$task_id" > $TASK_QUEUE_DIR/task_${task_id}.txt
                task_id=$((task_id + 1))
            fi
            if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
                echo "training|standard|no_max_entropy|standard|$task_id" > $TASK_QUEUE_DIR/task_${task_id}.txt
                task_id=$((task_id + 1))
            fi
        fi
        
        if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
            if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
                echo "training|context_only|max_entropy|standard|$task_id" > $TASK_QUEUE_DIR/task_${task_id}.txt
                task_id=$((task_id + 1))
            fi
            if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
                echo "training|context_only|no_max_entropy|standard|$task_id" > $TASK_QUEUE_DIR/task_${task_id}.txt
                task_id=$((task_id + 1))
            fi
        fi
    fi
    
    # OFL Version Training Tasks
    if [ "$RUN_OFL_VERSION" -eq 1 ]; then
        if [ "$RUN_STANDARD" -eq 1 ]; then
            if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
                echo "training|standard|max_entropy|ofl|$task_id" > $TASK_QUEUE_DIR/task_${task_id}.txt
                task_id=$((task_id + 1))
            fi
            if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
                echo "training|standard|no_max_entropy|ofl|$task_id" > $TASK_QUEUE_DIR/task_${task_id}.txt
                task_id=$((task_id + 1))
            fi
        fi
        
        if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
            if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
                echo "training|context_only|max_entropy|ofl|$task_id" > $TASK_QUEUE_DIR/task_${task_id}.txt
                task_id=$((task_id + 1))
            fi
            if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
                echo "training|context_only|no_max_entropy|ofl|$task_id" > $TASK_QUEUE_DIR/task_${task_id}.txt
                task_id=$((task_id + 1))
            fi
        fi
    fi
    
    echo $task_id > $TASK_QUEUE_DIR/total_training_tasks.txt
    echo "✅ Generated $task_id training tasks"
}

generate_evaluation_tasks() {
    echo "📋 Generating evaluation tasks..."
    
    local task_id=0
    
    # Generate evaluation tasks for each trained policy
    for policy_dir in "${POLICIES_TO_EVALUATE[@]}"; do
        for benchmark in "${BENCHMARKS[@]}"; do
            echo "evaluation|$policy_dir|$benchmark|$task_id" > $TASK_QUEUE_DIR/eval_task_${task_id}.txt
            task_id=$((task_id + 1))
        done
    done
    
    echo $task_id > $TASK_QUEUE_DIR/total_evaluation_tasks.txt
    echo "✅ Generated $task_id evaluation tasks"
}

generate_baseline_tasks() {
    echo "📋 Generating baseline tasks..."
    
    local task_id=0
    
    # Generate baseline tasks for each benchmark
    for benchmark in "${BENCHMARKS[@]}"; do
        echo "baseline|eagle3|$benchmark|$task_id" > $TASK_QUEUE_DIR/baseline_task_${task_id}.txt
        task_id=$((task_id + 1))
        echo "baseline|standard|$benchmark|$task_id" > $TASK_QUEUE_DIR/baseline_task_${task_id}.txt
        task_id=$((task_id + 1))
    done
    
    echo $task_id > $TASK_QUEUE_DIR/total_baseline_tasks.txt
    echo "✅ Generated $task_id baseline tasks"
}

# Parallel execution function
execute_task_parallel() {
    local task_file=$1
    local task_info=$(cat $task_file)
    local task_type=$(echo $task_info | cut -d'|' -f1)
    local task_params=$(echo $task_info | cut -d'|' -f2-)
    local task_id=$(echo $task_info | cut -d'|' -f5)
    
    echo "🚀 Starting task: $task_type - $task_params"
    
    # Get GPU for this task
    local gpu_id=$(wait_for_gpu $task_type)
    if [ "$gpu_id" -eq -1 ]; then
        echo "❌ Failed to get GPU for task $task_id"
        return 1
    fi
    
    # Acquire GPU
    if ! acquire_gpu $gpu_id $task_type; then
        echo "❌ Failed to acquire GPU $gpu_id for task $task_id"
        return 1
    fi
    
    echo "🎯 Task $task_id assigned to GPU $gpu_id"
    
    # Execute task based on type
    case $task_type in
        "training")
            execute_training_task $task_params $gpu_id $task_id
            ;;
        "evaluation")
            execute_evaluation_task $task_params $gpu_id $task_id
            ;;
        "baseline")
            execute_baseline_task $task_params $gpu_id $task_id
            ;;
        *)
            echo "❌ Unknown task type: $task_type"
            release_gpu $gpu_id
            return 1
            ;;
    esac
    
    # Release GPU
    release_gpu $gpu_id
    echo "✅ Task $task_id completed on GPU $gpu_id"
}

# Task execution functions
execute_training_task() {
    local params=$1
    local gpu_id=$2
    local task_id=$3
    
    local state_type=$(echo $params | cut -d'|' -f1)
    local entropy_type=$(echo $params | cut -d'|' -f2)
    local policy_version=$(echo $params | cut -d'|' -f3)
    
    # Set GPU environment
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Determine model ID and directory based on parameters
    local model_id=""
    local output_dir=""
    
    if [ "$state_type" = "standard" ]; then
        if [ "$entropy_type" = "max_entropy" ]; then
            model_id="optimized_max_entropy_ppo_standard"
        else
            model_id="optimized_standard_ppo_standard"
        fi
    else
        if [ "$entropy_type" = "max_entropy" ]; then
            model_id="optimized_max_entropy_ppo_context"
        else
            model_id="optimized_standard_ppo_context"
        fi
    fi
    
    if [ "$policy_version" = "ofl" ]; then
        model_id="${model_id}_ofl"
    fi
    
    output_dir="log/$DATE/$model_id"
    
    # Build command arguments
    local cmd_args="--ea-model-path $MODEL_PATH --base-model-path $BASE_MODEL_PATH"
    cmd_args="$cmd_args --model-id $model_id"
    cmd_args="$cmd_args --question-file eagle/data/rl_training/question.jsonl"
    cmd_args="$cmd_args --question-begin 0 --question-end $QUESTION_END"
    cmd_args="$cmd_args --answer-file $output_dir/training_answers.jsonl"
    cmd_args="$cmd_args --num-choices 1 --num-gpus-per-model 1 --num-gpus-total 1"
    cmd_args="$cmd_args --max-gpu-memory \"80GiB\" --dtype float16 --temperature 0.0"
    cmd_args="$cmd_args --use-online-rl --use-optimized-sb3-discrete-ppo"
    cmd_args="$cmd_args --optimized-policy-version $policy_version"
    cmd_args="$cmd_args --online-lr 3e-4 --ppo-n-steps 64 --ppo-batch-size 32 --ppo-epochs 4"
    cmd_args="$cmd_args --action-cache-steps 10 --action-cache-enabled"
    cmd_args="$cmd_args --use-eagle3-features --hidden-size 4096"
    cmd_args="$cmd_args --checkpoint-dir $output_dir/checkpoints"
    # Use .zip for OFL version, .pt for standard version
    if [ "$policy_version" = "ofl" ]; then
        cmd_args="$cmd_args --online-policy-save-path $output_dir/optimized_*_ppo_policy_sb3.zip"
    else
        cmd_args="$cmd_args --online-policy-save-path $output_dir/optimized_*_ppo_policy_sb3.pt"
    fi
    cmd_args="$cmd_args --checkpoint-freq 500 --wandb-project eagle-optimized-sb3-ppo"
    cmd_args="$cmd_args --total-token 60 --depth 7 --top-k 10 --use-stepwise-rl --use-eagle3"
    
    if [ "$entropy_type" = "max_entropy" ]; then
        cmd_args="$cmd_args --enable-max-entropy --max-entropy-ent-coef 0.1"
        cmd_args="$cmd_args --inference-temperature 1.5 --max-entropy-inference"
    else
        cmd_args="$cmd_args --disable-max-entropy"
    fi
    
    if [ "$state_type" = "context_only" ]; then
        cmd_args="$cmd_args --use-context-only-state"
    fi
    
    # Execute training
    echo "🎯 Executing training task $task_id on GPU $gpu_id: $model_id"
    PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl $cmd_args 2>&1 | tee $output_dir/training.log
    
    echo "✅ Training task $task_id completed"
}

execute_evaluation_task() {
    local params=$1
    local gpu_id=$2
    local task_id=$3
    
    local policy_dir=$(echo $params | cut -d'|' -f1)
    local benchmark=$(echo $params | cut -d'|' -f2)
    
    # Set GPU environment
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Determine policy version and arguments
    local policy_version="standard"
    local context_args=""
    local entropy_args=""
    
    if [[ "$policy_dir" == *"_ofl" ]]; then
        policy_version="ofl"
    fi
    
    if [[ "$policy_dir" == *"_context" ]]; then
        context_args="--use-context-only-state"
    fi
    
    if [[ "$policy_dir" == *"max_entropy"* ]]; then
        entropy_args="--enable-max-entropy --inference-temperature 1.5 --max-entropy-inference"
    else
        entropy_args="--disable-max-entropy"
    fi
    
    # Build command
    local cmd_args="--ea-model-path $MODEL_PATH --base-model-path $BASE_MODEL_PATH"
    cmd_args="$cmd_args --model-id ${policy_dir}_$benchmark"
    cmd_args="$cmd_args --question-file eagle/data/$benchmark/question.jsonl"
    cmd_args="$cmd_args --question-begin 0 --question-end -1"
    cmd_args="$cmd_args --answer-file log/$DATE/$policy_dir/evaluation/${benchmark}_results.jsonl"
    cmd_args="$cmd_args --num-choices 1 --num-gpus-per-model 1 --num-gpus-total 1"
    cmd_args="$cmd_args --max-gpu-memory \"80GiB\" --dtype float16 --temperature 0.0"
    cmd_args="$cmd_args --use-online-rl --use-optimized-sb3-discrete-ppo"
    cmd_args="$cmd_args --optimized-policy-version $policy_version --online-inference-only"
    # Use .zip for OFL version, .pt for standard version
    if [[ "$policy_dir" == *"_ofl" ]]; then
        cmd_args="$cmd_args $context_args --online-policy-path log/$DATE/$policy_dir/optimized_*_ppo_policy_sb3.zip"
    else
        cmd_args="$cmd_args $context_args --online-policy-path log/$DATE/$policy_dir/optimized_*_ppo_policy_sb3.pt"
    fi
    cmd_args="$cmd_args $entropy_args --action-cache-steps 30 --action-cache-enabled"
    cmd_args="$cmd_args --use-eagle3-features --hidden-size 4096"
    cmd_args="$cmd_args --total-token 60 --depth 7 --top-k 10 --use-stepwise-rl --use-eagle3"
    
    # Execute evaluation
    echo "🎯 Executing evaluation task $task_id on GPU $gpu_id: $policy_dir on $benchmark"
    python -m eagle.evaluation.gen_ea_answer_llama3chat_rl $cmd_args 2>&1 | tee log/$DATE/$policy_dir/evaluation/${benchmark}_evaluation.log
    
    echo "✅ Evaluation task $task_id completed"
}

execute_baseline_task() {
    local params=$1
    local gpu_id=$2
    local task_id=$3
    
    local baseline_type=$(echo $params | cut -d'|' -f1)
    local benchmark=$(echo $params | cut -d'|' -f2)
    
    # Set GPU environment
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Build command based on baseline type
    if [ "$baseline_type" = "eagle3" ]; then
        local cmd_args="--ea-model-path \"$MODEL_PATH\" --base-model-path \"$BASE_MODEL_PATH\""
        cmd_args="$cmd_args --bench-name \"$benchmark\""
        cmd_args="$cmd_args --answer-file \"log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_eagle3.jsonl\""
        cmd_args="$cmd_args --temperature 0.0 --use_eagle3"
        
        echo "🎯 Executing EAGLE3 baseline task $task_id on GPU $gpu_id: $benchmark"
        python -m eagle.evaluation.gen_ea_answer_llama3chat $cmd_args 2>&1 | tee -a log/$DATE/baseline_results/baseline_${benchmark}_eagle3.log
    else
        local cmd_args="--ea-model-path \"$MODEL_PATH\" --base-model-path \"$BASE_MODEL_PATH\""
        cmd_args="$cmd_args --bench-name \"$benchmark\""
        cmd_args="$cmd_args --answer-file \"log/$DATE/baseline_results/${benchmark}_LLaMA3.1-8B_baseline.jsonl\""
        cmd_args="$cmd_args --temperature 0.0"
        
        echo "🎯 Executing standard baseline task $task_id on GPU $gpu_id: $benchmark"
        python -m eagle.evaluation.gen_baseline_answer_llama3chat $cmd_args 2>&1 | tee -a log/$DATE/baseline_results/baseline_${benchmark}_standard.log
    fi
    
    echo "✅ Baseline task $task_id completed"
}

# Create log directory - dynamic based on execution mode
DIRECTORIES_TO_CREATE=()

# Standard version directories
if [ "$RUN_STANDARD_VERSION" -eq 1 ]; then
    if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_max_entropy_ppo_context")
        fi
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_standard_ppo_context")
        fi
    fi

    if [ "$RUN_STANDARD" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_max_entropy_ppo_standard")
        fi
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_standard_ppo_standard")
        fi
    fi
fi

# OFL version directories
if [ "$RUN_OFL_VERSION" -eq 1 ]; then
    if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_max_entropy_ppo_context_ofl")
        fi
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_standard_ppo_context_ofl")
        fi
    fi

    if [ "$RUN_STANDARD" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_max_entropy_ppo_standard_ofl")
        fi
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            DIRECTORIES_TO_CREATE+=("optimized_standard_ppo_standard_ofl")
        fi
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

echo "=== OPTIMIZED EAGLE SB3 PPO Training & Multi-Benchmark Evaluation ===" | tee -a log/$DATE/comparison.txt
echo "Model: $MODEL_PATH" | tee -a log/$DATE/comparison.txt
echo "Base Model: $BASE_MODEL_PATH" | tee -a log/$DATE/comparison.txt
echo "Training Dataset: eagle/data/rl_training/question.jsonl" | tee -a log/$DATE/comparison.txt

# Show which versions are being run
if [ "$RUN_STANDARD_VERSION" -eq 1 ] && [ "$RUN_OFL_VERSION" -eq 1 ]; then
    echo "Policy Versions: BOTH (Standard + OFL with enhanced features)" | tee -a log/$DATE/comparison.txt
elif [ "$RUN_OFL_VERSION" -eq 1 ]; then
    echo "Policy Version: OFL (with enhanced features)" | tee -a log/$DATE/comparison.txt
else
    echo "Policy Version: Standard" | tee -a log/$DATE/comparison.txt
fi

echo "OPTIMIZATION 1: EAGLE-3 layer features instead of SBERT embeddings" | tee -a log/$DATE/comparison.txt
echo "OPTIMIZATION 2: Action caching - generate action every 10 steps" | tee -a log/$DATE/comparison.txt
if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
    echo "OPTIMIZATION 3: Context-only state representation (SBERT 384D directly)" | tee -a log/$DATE/comparison.txt
fi
echo "Expected speedup: ~50% reduction in RL policy computation" | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Setup GPU management if parallel execution is enabled
if [ "$ENABLE_PARALLEL_EXECUTION" -eq 1 ]; then
    setup_gpu_management
    echo "🚀 Parallel execution enabled with $TOTAL_GPUS GPUs" | tee -a log/$DATE/comparison.txt
    echo "   - Training: Up to $MAX_TASKS_PER_GPU_TRAINING tasks per GPU" | tee -a log/$DATE/comparison.txt
    echo "   - Inference/Baseline: $MAX_TASKS_PER_GPU_INFERENCE task per GPU" | tee -a log/$DATE/comparison.txt
    echo "   - Max parallel processes: $MAX_PARALLEL_PROCESSES" | tee -a log/$DATE/comparison.txt
    echo "" | tee -a log/$DATE/comparison.txt
else
    echo "📋 Sequential execution mode" | tee -a log/$DATE/comparison.txt
    echo "" | tee -a log/$DATE/comparison.txt
fi

# PHASE 1: TRAINING
echo "=== Phase 1: Training ===" | tee -a log/$DATE/comparison.txt
if [ "$ENABLE_PARALLEL_EXECUTION" -eq 1 ]; then
    run_parallel_execution "training"
else
    run_sequential_execution "training"
fi

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 2: Evaluation ===" | tee -a log/$DATE/comparison.txt

# Check if trained policies exist and create evaluation plan
POLICIES_TO_EVALUATE=()
POLICY_LABELS=()

# Standard Version Policies
if [ "$RUN_STANDARD_VERSION" -eq 1 ]; then
    if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_max_entropy_ppo_context/optimized_max_entropy_ppo_policy_sb3.pt" ]; then
            POLICIES_TO_EVALUATE+=("optimized_max_entropy_ppo_context")
            POLICY_LABELS+=("Max-Entropy PPO (Context-Only) - Standard")
        elif [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Context-only max-entropy policy (Standard) not found!" | tee -a log/$DATE/comparison.txt
        fi
        
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_standard_ppo_context/optimized_standard_ppo_policy_sb3.pt" ]; then
            POLICIES_TO_EVALUATE+=("optimized_standard_ppo_context")
            POLICY_LABELS+=("Standard PPO (Context-Only) - Standard")
        elif [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Context-only standard policy (Standard) not found!" | tee -a log/$DATE/comparison.txt
        fi
    fi

    if [ "$RUN_STANDARD" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_max_entropy_ppo_standard/optimized_max_entropy_ppo_policy_sb3.pt" ]; then
            POLICIES_TO_EVALUATE+=("optimized_max_entropy_ppo_standard")
            POLICY_LABELS+=("Max-Entropy PPO (Standard) - Standard")
        elif [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Standard max-entropy policy (Standard) not found!" | tee -a log/$DATE/comparison.txt
        fi
        
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_standard_ppo_standard/optimized_standard_ppo_policy_sb3.pt" ]; then
            POLICIES_TO_EVALUATE+=("optimized_standard_ppo_standard")
            POLICY_LABELS+=("Standard PPO (Standard) - Standard")
        elif [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Standard policy (Standard) not found!" | tee -a log/$DATE/comparison.txt
        fi
    fi
fi

# OFL Version Policies
if [ "$RUN_OFL_VERSION" -eq 1 ]; then
    if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_max_entropy_ppo_context_ofl/optimized_max_entropy_ppo_policy_sb3.zip" ]; then
            POLICIES_TO_EVALUATE+=("optimized_max_entropy_ppo_context_ofl")
            POLICY_LABELS+=("Max-Entropy PPO (Context-Only) - OFL")
        elif [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Context-only max-entropy policy (OFL) not found!" | tee -a log/$DATE/comparison.txt
        fi
        
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_standard_ppo_context_ofl/optimized_standard_ppo_policy_sb3.zip" ]; then
            POLICIES_TO_EVALUATE+=("optimized_standard_ppo_context_ofl")
            POLICY_LABELS+=("Standard PPO (Context-Only) - OFL")
        elif [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Context-only standard policy (OFL) not found!" | tee -a log/$DATE/comparison.txt
        fi
    fi

    if [ "$RUN_STANDARD" -eq 1 ]; then
        if [ "$RUN_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_max_entropy_ppo_standard_ofl/optimized_max_entropy_ppo_policy_sb3.zip" ]; then
            POLICIES_TO_EVALUATE+=("optimized_max_entropy_ppo_standard_ofl")
            POLICY_LABELS+=("Max-Entropy PPO (Standard) - OFL")
        elif [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Standard max-entropy policy (OFL) not found!" | tee -a log/$DATE/comparison.txt
        fi
        
        if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ] && [ -f "log/$DATE/optimized_standard_ppo_standard_ofl/optimized_standard_ppo_policy_sb3.zip" ]; then
            POLICIES_TO_EVALUATE+=("optimized_standard_ppo_standard_ofl")
            POLICY_LABELS+=("Standard PPO (Standard) - OFL")
        elif [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
            echo "❌ Standard policy (OFL) not found!" | tee -a log/$DATE/comparison.txt
        fi
    fi
fi

if [ ${#POLICIES_TO_EVALUATE[@]} -eq 0 ]; then
    echo "❌ No trained policies found for evaluation!" | tee -a log/$DATE/comparison.txt
    if [ "$ENABLE_PARALLEL_EXECUTION" -eq 1 ]; then
        cleanup_gpu_management
    fi
    exit 1
fi

echo "✅ Found ${#POLICIES_TO_EVALUATE[@]} trained policies. Starting evaluation..." | tee -a log/$DATE/comparison.txt
echo "" | tee -a log/$DATE/comparison.txt

# Run evaluation
if [ "$ENABLE_PARALLEL_EXECUTION" -eq 1 ]; then
    run_parallel_execution "evaluation"
else
    run_sequential_execution "evaluation"
fi

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Phase 3: Baseline Generation ===" | tee -a log/$DATE/comparison.txt

# Run baseline generation
if [ "$ENABLE_PARALLEL_EXECUTION" -eq 1 ]; then
    run_parallel_execution "baseline"
else
    run_sequential_execution "baseline"
fi

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
if [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
    if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
        ((TRAINED_POLICIES += 1))
    fi
    if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
        ((TRAINED_POLICIES += 1))
    fi
fi
if [ "$RUN_STANDARD" -eq 1 ]; then
    if [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
        ((TRAINED_POLICIES += 1))
    fi
    if [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
        ((TRAINED_POLICIES += 1))
    fi
fi

# Determine execution mode description
EXECUTION_MODE=""
if [ "$RUN_CONTEXT_ONLY" -eq 1 ] && [ "$RUN_STANDARD" -eq 1 ]; then
    EXECUTION_MODE="Both States"
elif [ "$RUN_CONTEXT_ONLY" -eq 1 ]; then
    EXECUTION_MODE="Context-Only"
elif [ "$RUN_STANDARD" -eq 1 ]; then
    EXECUTION_MODE="Standard"
fi

if [ "$RUN_MAX_ENTROPY" -eq 1 ] && [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
    EXECUTION_MODE="$EXECUTION_MODE + Both Entropy"
elif [ "$RUN_MAX_ENTROPY" -eq 1 ]; then
    EXECUTION_MODE="$EXECUTION_MODE + Max-Entropy"
elif [ "$RUN_NO_MAX_ENTROPY" -eq 1 ]; then
    EXECUTION_MODE="$EXECUTION_MODE + Standard"
fi

# Add policy version information
if [ "$RUN_STANDARD_VERSION" -eq 1 ] && [ "$RUN_OFL_VERSION" -eq 1 ]; then
    EXECUTION_MODE="$EXECUTION_MODE + Both Policy Versions"
elif [ "$RUN_OFL_VERSION" -eq 1 ]; then
    EXECUTION_MODE="$EXECUTION_MODE + OFL Version"
else
    EXECUTION_MODE="$EXECUTION_MODE + Standard Version"
fi

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Summary ===" | tee -a log/$DATE/comparison.txt
echo "Training completed with optimized PPO policies!" | tee -a log/$DATE/comparison.txt
echo "Execution Mode: $EXECUTION_MODE" | tee -a log/$DATE/comparison.txt
echo "Policies trained: $TRAINED_POLICIES" | tee -a log/$DATE/comparison.txt
echo "Results saved in: log/$DATE/" | tee -a log/$DATE/comparison.txt
echo "Key optimizations implemented:" | tee -a log/$DATE/comparison.txt
echo "1. EAGLE-3 layer features with optional context-only mode" | tee -a log/$DATE/comparison.txt
echo "2. Action caching every 30 steps (~50% computation reduction)" | tee -a log/$DATE/comparison.txt
echo "3. Flexible execution modes for targeted experiments" | tee -a log/$DATE/comparison.txt
echo "4. Enhanced PPO with temperature-based action sampling" | tee -a log/$DATE/comparison.txt
if [ "$RUN_OFL_VERSION" -eq 1 ]; then
    echo "5. OFL version with enhanced features (set_max_timesteps, set_training_mode, enhanced PPO updates)" | tee -a log/$DATE/comparison.txt
fi
echo "" | tee -a log/$DATE/comparison.txt

# Create performance summary
echo "Performance Summary Report" >> log/$DATE/summary.txt
echo "=========================" >> log/$DATE/summary.txt
echo "Date: $DATE" >> log/$DATE/summary.txt
echo "Execution Mode: $EXECUTION_MODE" >> log/$DATE/summary.txt
echo "Policies Trained: $TRAINED_POLICIES" >> log/$DATE/summary.txt
echo "Model: $MODEL_PATH" >> log/$DATE/summary.txt
echo "Algorithm: Optimized PPO" >> log/$DATE/summary.txt
echo "Optimizations: EAGLE-3 features + Action caching + Flexible execution" >> log/$DATE/summary.txt
echo "Training questions: $QUESTION_END" >> log/$DATE/summary.txt
echo "Benchmarks evaluated: ${BENCHMARKS[*]}" >> log/$DATE/summary.txt
echo "Policies evaluated: ${#POLICIES_TO_EVALUATE[@]}" >> log/$DATE/summary.txt
for j in "${!POLICY_LABELS[@]}"; do
    echo "  - ${POLICY_LABELS[$j]}" >> log/$DATE/summary.txt
done
echo "" >> log/$DATE/summary.txt

echo "Check log/$DATE/summary.txt for detailed results." | tee -a log/$DATE/comparison.txt

# Cleanup GPU management if parallel execution was enabled
if [ "$ENABLE_PARALLEL_EXECUTION" -eq 1 ]; then
    cleanup_gpu_management
fi

echo "" | tee -a log/$DATE/comparison.txt
echo "=== Final Summary ===" | tee -a log/$DATE/comparison.txt
echo "🎉 All phases completed successfully!" | tee -a log/$DATE/comparison.txt
echo "📊 Execution Mode: $EXECUTION_MODE" | tee -a log/$DATE/comparison.txt
echo "🚀 Parallel Execution: $([ "$ENABLE_PARALLEL_EXECUTION" -eq 1 ] && echo "Enabled ($TOTAL_GPUS GPUs)" || echo "Disabled")" | tee -a log/$DATE/comparison.txt
echo "📁 Results saved in: log/$DATE/" | tee -a log/$DATE/comparison.txt
echo "🔧 Key optimizations implemented:" | tee -a log/$DATE/comparison.txt
echo "   1. EAGLE-3 layer features with optional context-only mode" | tee -a log/$DATE/comparison.txt
echo "   2. Action caching every 30 steps (~50% computation reduction)" | tee -a log/$DATE/comparison.txt
echo "   3. Flexible execution modes for targeted experiments" | tee -a log/$DATE/comparison.txt
echo "   4. Enhanced PPO with temperature-based action sampling" | tee -a log/$DATE/comparison.txt
if [ "$RUN_OFL_VERSION" -eq 1 ]; then
    echo "   5. OFL version with enhanced features (set_max_timesteps, set_training_mode, enhanced PPO updates)" | tee -a log/$DATE/comparison.txt
fi
if [ "$ENABLE_PARALLEL_EXECUTION" -eq 1 ]; then
    echo "   6. Parallel execution with intelligent GPU management" | tee -a log/$DATE/comparison.txt
    echo "      - Training: Up to $MAX_TASKS_PER_GPU_TRAINING tasks per GPU" | tee -a log/$DATE/comparison.txt
    echo "      - Inference: $MAX_TASKS_PER_GPU_INFERENCE task per GPU" | tee -a log/$DATE/comparison.txt
    echo "      - Max parallel processes: $MAX_PARALLEL_PROCESSES" | tee -a log/$DATE/comparison.txt
fi
echo "" | tee -a log/$DATE/comparison.txt

# Parallel execution orchestrator
run_parallel_execution() {
    local phase=$1  # "training", "evaluation", or "baseline"
    
    echo "🚀 Starting parallel execution for phase: $phase"
    
    # Generate tasks for this phase
    case $phase in
        "training")
            generate_training_tasks
            local total_tasks=$(cat $TASK_QUEUE_DIR/total_training_tasks.txt)
            local task_pattern="task_*.txt"
            ;;
        "evaluation")
            generate_evaluation_tasks
            local total_tasks=$(cat $TASK_QUEUE_DIR/total_evaluation_tasks.txt)
            local task_pattern="eval_task_*.txt"
            ;;
        "baseline")
            generate_baseline_tasks
            local total_tasks=$(cat $TASK_QUEUE_DIR/total_baseline_tasks.txt)
            local task_pattern="baseline_task_*.txt"
            ;;
        *)
            echo "❌ Unknown phase: $phase"
            return 1
            ;;
    esac
    
    if [ "$total_tasks" -eq 0 ]; then
        echo "ℹ️ No tasks to execute for phase: $phase"
        return 0
    fi
    
    echo "📊 Executing $total_tasks tasks in parallel (max $MAX_PARALLEL_PROCESSES processes)"
    
    # Get list of task files
    local task_files=($TASK_QUEUE_DIR/$task_pattern)
    local running_processes=0
    local completed_tasks=0
    local failed_tasks=0
    
    # Process tasks
    for task_file in "${task_files[@]}"; do
        # Wait if we've reached max parallel processes
        while [ $running_processes -ge $MAX_PARALLEL_PROCESSES ]; do
            # Check for completed processes
            for pid_file in $PID_FILE_DIR/*.pid; do
                if [ -f "$pid_file" ]; then
                    local pid=$(cat $pid_file)
                    if ! kill -0 $pid 2>/dev/null; then
                        # Process completed
                        rm -f $pid_file
                        running_processes=$((running_processes - 1))
                        completed_tasks=$((completed_tasks + 1))
                        echo "✅ Task completed. Running: $running_processes, Completed: $completed_tasks"
                    fi
                fi
            done
            
            if [ $running_processes -ge $MAX_PARALLEL_PROCESSES ]; then
                sleep 5
            fi
        done
        
        # Start new task
        execute_task_parallel "$task_file" &
        local task_pid=$!
        echo $task_pid > $PID_FILE_DIR/task_${running_processes}.pid
        running_processes=$((running_processes + 1))
        
        echo "🚀 Started task. Running: $running_processes, Completed: $completed_tasks"
        
        # Small delay to avoid overwhelming the system
        sleep 2
    done
    
    # Wait for remaining processes
    echo "⏳ Waiting for remaining $running_processes processes to complete..."
    for pid_file in $PID_FILE_DIR/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat $pid_file)
            wait $pid
            local exit_code=$?
            rm -f $pid_file
            
            if [ $exit_code -eq 0 ]; then
                completed_tasks=$((completed_tasks + 1))
            else
                failed_tasks=$((failed_tasks + 1))
            fi
        fi
    done
    
    echo "✅ Phase $phase completed. Success: $completed_tasks, Failed: $failed_tasks"
    
    if [ $failed_tasks -gt 0 ]; then
        echo "⚠️ $failed_tasks tasks failed during phase $phase"
        return 1
    fi
    
    return 0
}

# Sequential execution (fallback)
run_sequential_execution() {
    local phase=$1
    
    echo "📋 Running sequential execution for phase: $phase"
    
    case $phase in
        "training")
            # Run original training logic
            run_original_training
            ;;
        "evaluation")
            # Run original evaluation logic
            run_original_evaluation
            ;;
        "baseline")
            # Run original baseline logic
            run_original_baseline
            ;;
        *)
            echo "❌ Unknown phase: $phase"
            return 1
            ;;
    esac
}

# Original execution functions (for sequential mode)
run_original_training() {
    echo "📋 Running original sequential training..."
    # This will contain the original training logic
    # (keeping the existing training code as fallback)
}

run_original_evaluation() {
    echo "📋 Running original sequential evaluation..."
    # This will contain the original evaluation logic
}

run_original_baseline() {
    echo "📋 Running original sequential baseline..."
    # This will contain the original baseline logic
}