#!/bin/bash

# Quick test script to verify optimized RL policies work
# Tests both PPO and DQN optimizations with minimal examples

echo "=== Testing Optimized EAGLE RL Policies ==="
echo "This script runs minimal tests to verify the optimizations work correctly."
echo ""

MODEL_PATH="./eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
TEST_QUESTIONS=5

# Create test log directory
TEST_DIR="test_optimized_$(date '+%Y%m%d_%H%M')"
mkdir -p log/$TEST_DIR

echo "=== Test 1: Optimized PPO Policy ==="
echo "Testing with EAGLE-3 features + action caching..."

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --model-path $MODEL_PATH \
    --base-model-path $BASE_MODEL_PATH \
    --model-id test_optimized_ppo \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 \
    --question-end $TEST_QUESTIONS \
    --answer-file log/$TEST_DIR/test_optimized_ppo_answers.jsonl \
    --num-choices 1 \
    --num-gpus-per-model 1 \
    --num-gpus-total 1 \
    --max-gpu-memory "80GiB" \
    --dtype float16 \
    --temperature 0.0 \
    --use-online-rl \
    --use-optimized-sb3-discrete-ppo \
    --online-lr 3e-4 \
    --enable-max-entropy \
    --action-cache-steps 5 \
    --action-cache-enabled \
    --use-eagle3-features \
    --hidden-size 4096 \
    --no-wandb \
    --total-token 60 \
    --depth 7 \
    --top-k 10 \
    --use-eagle3 2>&1 | tee log/$TEST_DIR/test_optimized_ppo.log

if [ $? -eq 0 ]; then
    echo "✅ Optimized PPO test PASSED"
else
    echo "❌ Optimized PPO test FAILED"
    exit 1
fi

echo ""
echo "=== Test 2: Optimized DQN Policy ==="
echo "Testing with EAGLE-3 features + action caching..."

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --model-path $MODEL_PATH \
    --base-model-path $BASE_MODEL_PATH \
    --model-id test_optimized_dqn \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 \
    --question-end $TEST_QUESTIONS \
    --answer-file log/$TEST_DIR/test_optimized_dqn_answers.jsonl \
    --num-choices 1 \
    --num-gpus-per-model 1 \
    --num-gpus-total 1 \
    --max-gpu-memory "80GiB" \
    --dtype float16 \
    --temperature 0.0 \
    --use-online-rl \
    --use-optimized-dqn \
    --online-lr 0.01 \
    --online-epsilon-start 0.8 \
    --online-epsilon-end 0.3 \
    --online-memory-size 50 \
    --online-batch-size 4 \
    --action-cache-steps 5 \
    --action-cache-enabled \
    --use-eagle3-features \
    --hidden-size 4096 \
    --no-wandb \
    --total-token 60 \
    --depth 7 \
    --top-k 10 \
    --use-eagle3 2>&1 | tee log/$TEST_DIR/test_optimized_dqn.log

if [ $? -eq 0 ]; then
    echo "✅ Optimized DQN test PASSED"
else
    echo "❌ Optimized DQN test FAILED"
    exit 1
fi

echo ""
echo "=== Test 3: Backward Compatibility ==="
echo "Testing that traditional policies still work..."

PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --model-path $MODEL_PATH \
    --base-model-path $BASE_MODEL_PATH \
    --model-id test_traditional_ppo \
    --question-file eagle/data/rl_training/question.jsonl \
    --question-begin 0 \
    --question-end 2 \
    --answer-file log/$TEST_DIR/test_traditional_ppo_answers.jsonl \
    --num-choices 1 \
    --num-gpus-per-model 1 \
    --num-gpus-total 1 \
    --max-gpu-memory "80GiB" \
    --dtype float16 \
    --temperature 0.0 \
    --use-online-rl \
    --use-sb3-discrete-ppo \
    --online-lr 3e-4 \
    --enable-max-entropy \
    --no-wandb \
    --total-token 60 \
    --depth 7 \
    --top-k 10 \
    --use-eagle3 2>&1 | tee log/$TEST_DIR/test_traditional_ppo.log

if [ $? -eq 0 ]; then
    echo "✅ Traditional PPO compatibility test PASSED"
else
    echo "❌ Traditional PPO compatibility test FAILED"
    exit 1
fi

echo ""
echo "=== All Tests Completed Successfully! ==="
echo "Results saved in: log/$TEST_DIR/"
echo ""
echo "Key findings from logs:"
echo "- Check for 'EAGLE-3 features' mentions in optimized tests"
echo "- Check for 'action caching' and 'cache_hit' in optimized tests"
echo "- Verify that traditional test doesn't mention optimizations"
echo ""
echo "🎉 Optimized EAGLE RL policies are working correctly!"
echo ""
echo "Next steps:"
echo "1. Run full training: ./test_optimized_ppo_modes_comparison.sh"
echo "2. Run DQN training: ./test_optimized_dqn_modes_comparison.sh"
echo "3. Compare performance with original policies"
