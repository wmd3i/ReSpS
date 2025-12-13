#!/bin/bash

# Complete pipeline for using ShareGPT and UltraChat-200K with EAGLE
# This script demonstrates the full workflow from data download to training

set -e

# Configuration
BASE_DIR="."
DATA_DIR="$BASE_DIR/training_data"
SCRIPTS_DIR="$BASE_DIR/download_dataset_scripts"
MODEL_PATH="$BASE_DIR/eagle_models/yuhuili_EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

echo "=== EAGLE Training Data Pipeline ==="
echo "Setting up directories..."
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/raw"
mkdir -p "$DATA_DIR/processed"

echo ""
echo "=== Step 1: Download Datasets ==="

# Note: These are example download commands - adjust based on actual dataset locations
echo "Download ShareGPT dataset..."
echo "  huggingface-cli download --repo-type dataset anon8231489123/ShareGPT_Vicuna_unfiltered --local-dir $DATA_DIR/raw/sharegpt"

huggingface-cli download --repo-type dataset anon8231489123/ShareGPT_Vicuna_unfiltered --local-dir $DATA_DIR/raw/sharegpt

echo "Download UltraChat-200K dataset..."  
echo "  huggingface-cli download --repo-type dataset HuggingFaceH4/ultrachat_200k --local-dir $DATA_DIR/raw/ultrachat"

huggingface-cli download --repo-type dataset HuggingFaceH4/ultrachat_200k --local-dir $DATA_DIR/raw/ultrachat

echo ""
echo "=== Step 2: Convert to EAGLE Format ==="

echo "Converting ShareGPT to EAGLE format..."
if [ -f "$DATA_DIR/raw/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json" ]; then
    python "$SCRIPTS_DIR/convert_sharegpt_to_eagle.py" \
        --input "$DATA_DIR/raw/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json" \
        --output "$DATA_DIR/processed/eagle_sharegpt.jsonl"
        # --max-conversations 50000
    echo "✓ ShareGPT conversion complete"
else
    echo "⚠ ShareGPT raw data not found at expected location"
fi

echo ""
echo "Converting UltraChat-200K to EAGLE format..."
if [ -d "$DATA_DIR/raw/ultrachat/data" ] && [ -n "$(ls -A $DATA_DIR/raw/ultrachat/data/train_sft*.parquet 2>/dev/null)" ]; then
    python "$SCRIPTS_DIR/convert_ultrachat_parquet_to_eagle.py" \
        --input-dir "$DATA_DIR/raw/ultrachat/data" \
        --output "$DATA_DIR/processed/eagle_ultrachat.jsonl" \
        --file-pattern "train_sft*.parquet"
        # --max-conversations 50000
    echo "✓ UltraChat conversion complete"
elif [ -f "$DATA_DIR/raw/ultrachat/train_sft.jsonl" ]; then
    python "$SCRIPTS_DIR/convert_ultrachat_to_eagle.py" \
        --input "$DATA_DIR/raw/ultrachat/train_sft.jsonl" \
        --output "$DATA_DIR/processed/eagle_ultrachat.jsonl"
        # --max-conversations 50000
    echo "✓ UltraChat conversion complete"
else
    echo "⚠ UltraChat raw data not found at expected location"
    echo "  Expected: $DATA_DIR/raw/ultrachat/data/train_sft*.parquet or $DATA_DIR/raw/ultrachat/train_sft.jsonl"
fi

echo ""
echo "=== Step 3: Data Validation ==="

echo "Validating ShareGPT data..."
if [ -f "$DATA_DIR/processed/eagle_sharegpt.jsonl" ]; then
    python "$SCRIPTS_DIR/validate_eagle_data.py" \
        --input "$DATA_DIR/processed/eagle_sharegpt.jsonl"
fi

echo ""
echo "Validating UltraChat data..."
if [ -f "$DATA_DIR/processed/eagle_ultrachat.jsonl" ]; then
    python "$SCRIPTS_DIR/validate_eagle_data.py" \
        --input "$DATA_DIR/processed/eagle_ultrachat.jsonl"
fi

echo ""
echo "=== Step 4: Combine Datasets (Optional) ==="

echo "Combining ShareGPT and UltraChat datasets..."
COMBINED_FILE="$DATA_DIR/processed/eagle_combined.jsonl"
if [ -f "$DATA_DIR/processed/eagle_sharegpt.jsonl" ] && [ -f "$DATA_DIR/processed/eagle_ultrachat.jsonl" ]; then
    cat "$DATA_DIR/processed/eagle_sharegpt.jsonl" "$DATA_DIR/processed/eagle_ultrachat.jsonl" > "$COMBINED_FILE"
    echo "✓ Combined dataset created: $COMBINED_FILE"
    
    # Validate combined dataset
    python "$SCRIPTS_DIR/validate_eagle_data.py" --input "$COMBINED_FILE"
else
    echo "⚠ Some datasets missing, skipping combination"
fi

echo "=== Next Steps ==="
echo "1. Use with Online RL:"
echo "   python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \\"
echo "     --base-model-path $BASE_MODEL_PATH \\"
echo "     --ea-model-path ./eagle_conversational_model \\"
echo "     --use-online-rl \\"
echo "     --online-lr 0.001"
echo ""
echo "Data processing complete! ✅"
