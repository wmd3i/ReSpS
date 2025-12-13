#!/bin/bash

# EAGLE RL Training with Combined Conversational Dataset
# This script converts the combined dataset to question format and runs RL training

set -e  # Exit on any error

# Configuration
COMBINED_DATASET="training_data/processed/eagle_combined.jsonl"
QUESTION_OUTPUT="eagle/data/rl_training/question.jsonl"
MODEL_PATH="eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# Create log directory with timestamp
DATE=$(date '+%Y%m%d_%H%M')
LOG_DIR="log/rl_training_$DATE"
mkdir -p "$LOG_DIR"

echo "=== EAGLE RL Training with Combined Dataset ===" | tee "$LOG_DIR/training.log"
echo "Combined Dataset: $COMBINED_DATASET" | tee -a "$LOG_DIR/training.log"
echo "Question Output: $QUESTION_OUTPUT" | tee -a "$LOG_DIR/training.log"
echo "Model: $MODEL_PATH" | tee -a "$LOG_DIR/training.log"
echo "Base Model: $BASE_MODEL_PATH" | tee -a "$LOG_DIR/training.log"
echo "Log Directory: $LOG_DIR" | tee -a "$LOG_DIR/training.log"
echo "" | tee -a "$LOG_DIR/training.log"

# Step 1: Check if combined dataset exists
if [ ! -f "$COMBINED_DATASET" ]; then
    echo "❌ Error: Combined dataset not found at $COMBINED_DATASET" | tee -a "$LOG_DIR/training.log"
    echo "Please run the dataset preparation script first:" | tee -a "$LOG_DIR/training.log"
    echo "  bash download_dataset_scripts/setup_conversational_training.sh" | tee -a "$LOG_DIR/training.log"
    exit 1
fi

echo "✅ Combined dataset found: $(wc -l < "$COMBINED_DATASET") lines" | tee -a "$LOG_DIR/training.log"

# Step 2: Convert combined dataset to question format
echo "Step 1: Converting combined dataset to MT-Bench question format..." | tee -a "$LOG_DIR/training.log"

python download_dataset_scripts/convert_combined_to_questions.py \
    --input "$COMBINED_DATASET" \
    --output "$QUESTION_OUTPUT" \
    --shuffle \
    --seed 42 \
    2>&1 | tee -a "$LOG_DIR/training.log"

if [ ! -f "$QUESTION_OUTPUT" ]; then
    echo "❌ Error: Question conversion failed" | tee -a "$LOG_DIR/training.log"
    exit 1
fi

# echo "✅ Question conversion complete: $(wc -l < "$QUESTION_OUTPUT") questions" | tee -a "$LOG_DIR/training.log"
# echo "" | tee -a "$LOG_DIR/training.log"

# # Step 3: Run RL training with the converted questions
# echo "Step 2: Starting RL training with max-entropy policy..." | tee -a "$LOG_DIR/training.log"

# PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
#     --base-model-path "$BASE_MODEL_PATH" \
#     --ea-model-path "$MODEL_PATH" \
#     --model-id "eagle-rl-conversational-$(date +%m%d_%H%M)" \
#     --question-file "$QUESTION_OUTPUT" \
#     --use-online-rl \
#     --online-lr 0.001 \
#     --online-epsilon-start 0.9 \
#     --online-epsilon-end 0.1 \
#     --online-memory-size 500 \
#     --online-batch-size 16 \
#     --online-policy-save-path "$LOG_DIR/conversational_rl_policy.pth" \
#     --temperature 0.0 \
#     --use_eagle3 \
#     --max-new-token 512 \
#     --num-choices 1 \
#     --checkpoint-dir "$LOG_DIR/checkpoints" \
#     --checkpoint-freq 50 \
#     --max-checkpoints 5 \
#     --online-repeat-factor 3 \
#     --training-seed 42 \
#     --wandb-project "eagle-conversational-rl" \
#     --answer-file "$LOG_DIR/training_results.jsonl" \
#     2>&1 | tee -a "$LOG_DIR/training.log"

# echo "" | tee -a "$LOG_DIR/training.log"

# # Step 4: Test the trained policy
# echo "Step 3: Testing trained policy with inference-only mode..." | tee -a "$LOG_DIR/training.log"

# # Create a smaller test set from mt_bench for quick evaluation
# python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
#     --base-model-path "$BASE_MODEL_PATH" \
#     --ea-model-path "$MODEL_PATH" \
#     --model-id "eagle-rl-test-$(date +%m%d_%H%M)" \
#     --bench-name "mt_bench" \
#     --use-online-rl \
#     --online-policy-path "$LOG_DIR/conversational_rl_policy.pth" \
#     --online-inference-only \
#     --temperature 0.0 \
#     --use_eagle3 \
#     --max-new-token 512 \
#     --num-choices 1 \
#     --answer-file "$LOG_DIR/test_results.jsonl" \
#     --question-begin 1 \
#     --question-end 20 \
#     2>&1 | tee -a "$LOG_DIR/training.log"

# echo "" | tee -a "$LOG_DIR/training.log"

# # Step 5: Performance analysis
# echo "Step 4: Performance analysis..." | tee -a "$LOG_DIR/training.log"

# echo "Training Results Summary:" | tee -a "$LOG_DIR/training.log"
# echo "=========================" | tee -a "$LOG_DIR/training.log"

# # Check if training results exist
# if [ -f "$LOG_DIR/training_results.jsonl" ]; then
#     TRAINING_LINES=$(wc -l < "$LOG_DIR/training_results.jsonl")
#     echo "✅ Training completed: $TRAINING_LINES questions processed" | tee -a "$LOG_DIR/training.log"
# else
#     echo "❌ Training results not found" | tee -a "$LOG_DIR/training.log"
# fi

# # Check if test results exist
# if [ -f "$LOG_DIR/test_results.jsonl" ]; then
#     TEST_LINES=$(wc -l < "$LOG_DIR/test_results.jsonl")
#     echo "✅ Testing completed: $TEST_LINES questions processed" | tee -a "$LOG_DIR/training.log"
# else
#     echo "❌ Test results not found" | tee -a "$LOG_DIR/training.log"
# fi

# # Check if policy was saved
# if [ -f "$LOG_DIR/conversational_rl_policy.pth" ]; then
#     POLICY_SIZE=$(du -h "$LOG_DIR/conversational_rl_policy.pth" | cut -f1)
#     echo "✅ Trained policy saved: $POLICY_SIZE" | tee -a "$LOG_DIR/training.log"
# else
#     echo "❌ Trained policy not found" | tee -a "$LOG_DIR/training.log"
# fi

# echo "" | tee -a "$LOG_DIR/training.log"
# echo "Files created:" | tee -a "$LOG_DIR/training.log"
# echo "- Question dataset: $QUESTION_OUTPUT" | tee -a "$LOG_DIR/training.log"
# echo "- Trained policy: $LOG_DIR/conversational_rl_policy.pth" | tee -a "$LOG_DIR/training.log"
# echo "- Training results: $LOG_DIR/training_results.jsonl" | tee -a "$LOG_DIR/training.log"
# echo "- Test results: $LOG_DIR/test_results.jsonl" | tee -a "$LOG_DIR/training.log"
# echo "- Training log: $LOG_DIR/training.log" | tee -a "$LOG_DIR/training.log"
# echo "- Checkpoints: $LOG_DIR/checkpoints/" | tee -a "$LOG_DIR/training.log"

# echo "" | tee -a "$LOG_DIR/training.log"
# echo "=== Next Steps ===" | tee -a "$LOG_DIR/training.log"
# echo "1. Check training metrics in wandb (if enabled)" | tee -a "$LOG_DIR/training.log"
# echo "2. Compare performance with baseline using speed.py" | tee -a "$LOG_DIR/training.log"
# echo "3. Use trained policy for inference:" | tee -a "$LOG_DIR/training.log"
# echo "   python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \\" | tee -a "$LOG_DIR/training.log"
# echo "     --use-online-rl \\" | tee -a "$LOG_DIR/training.log"
# echo "     --online-policy-path $LOG_DIR/conversational_rl_policy.pth \\" | tee -a "$LOG_DIR/training.log"
# echo "     --online-inference-only" | tee -a "$LOG_DIR/training.log"

# echo "" | tee -a "$LOG_DIR/training.log"
# echo "🎉 RL training pipeline complete!" | tee -a "$LOG_DIR/training.log"
