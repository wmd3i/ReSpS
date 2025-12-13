#!/bin/bash

# Convenience script for running EAGLE with RL policy
# This script demonstrates how to use RL-optimized evaluation without manually setting tree parameters

set -e

# Configuration - Update these paths for your setup
BASE_MODEL_PATH="/path/to/your/base/model"
EA_MODEL_PATH="/path/to/your/eagle/model"
RL_POLICY_PATH="ppo_tree_policy_refined.zip"
BENCH_NAME="mt_bench"
MODEL_ID="eagle-rl-auto"

# Check if RL policy exists
if [ ! -f "$RL_POLICY_PATH" ]; then
    echo "❌ RL policy not found at: $RL_POLICY_PATH"
    echo "Please train an RL policy first using:"
    echo "   python train_rl_policy.py --mode from_questions"
    exit 1
fi

echo "🚀 Running EAGLE with RL-optimized parameters"
echo "📊 Benchmark: $BENCH_NAME"
echo "🤖 RL Policy: $RL_POLICY_PATH"
echo "📝 Model ID: $MODEL_ID"
echo

# Run evaluation with RL policy
# Note: We don't specify --total-token, --depth, or --top-k because RL will predict them
python gen_ea_answer_llama3chat_rl.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --ea-model-path "$EA_MODEL_PATH" \
    --model-id "$MODEL_ID" \
    --bench-name "$BENCH_NAME" \
    --use-rl-policy \
    --rl-policy-path "$RL_POLICY_PATH" \
    --temperature 0.0 \
    --num-choices 1 \
    --max-new-token 1024

echo "✅ Evaluation completed!"
echo "📄 Results saved to: $BENCH_NAME/$MODEL_ID-temperature-0.0.jsonl"
echo
echo "🔍 To compare with fixed parameters, run:"
echo "python gen_ea_answer_llama3chat_rl.py \\"
echo "    --base-model-path \"$BASE_MODEL_PATH\" \\"
echo "    --ea-model-path \"$EA_MODEL_PATH\" \\"
echo "    --model-id \"eagle-fixed\" \\"
echo "    --bench-name \"$BENCH_NAME\" \\"
echo "    --total-token 60 --depth 5 --top-k 10"
