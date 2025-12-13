#!/bin/bash
# pip install "torch==2.0.1" "transformers==4.46.2" "accelerate==0.21.0"
# huggingface-cli logout
# huggingface-cli login
pip install huggingface_hub

# Create models directory
mkdir -p eagle_models
cd eagle_models

# EAGLE-3 Models
echo "Downloading EAGLE-3 models..."
models_eagle3=(
    "yuhuili/EAGLE3-Vicuna1.3-13B"
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
    "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"
    "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
    "nvidia/Llama-4-Maverick-17B-128E-Eagle3"
    "Tengyunw/qwen3_8b_eagle3"
    "Tengyunw/qwen3_30b_moe_eagle3"
    "AngelSlim/Qwen3-8B_eagle3"
    "AngelSlim/Qwen3-a3B_eagle3"
    "AngelSlim/Qwen3-4B_eagle3"
    "AngelSlim/Qwen3-32B_eagle3"
    "AngelSlim/Qwen3-1.7B_eagle3"
    "AngelSlim/Qwen3-14B_eagle3"
)

# EAGLE v1/v2 Models
echo "Downloading EAGLE v1/v2 models..."
models_eagle=(
    "yuhuili/EAGLE-Vicuna-7B-v1.3"
    "yuhuili/EAGLE-Vicuna-13B-v1.3"
    "yuhuili/EAGLE-Vicuna-33B-v1.3"
    "yuhuili/EAGLE-llama2-chat-7B"
    "yuhuili/EAGLE-llama2-chat-13B"
    "yuhuili/EAGLE-llama2-chat-70B"
    "yuhuili/EAGLE-mixtral-instruct-8x7B"
    "yuhuili/EAGLE-LLaMA3-Instruct-8B"
    "yuhuili/EAGLE-LLaMA3-Instruct-70B"
    "yuhuili/EAGLE-Qwen2-7B-Instruct"
    "yuhuili/EAGLE-Qwen2-72B-Instruct"
    "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
    "Zjcxy-SmartAI/Eagle-Qwen2.5-14B-Instruct"
)

# Download EAGLE-3 models
for model in "${models_eagle3[@]}"; do
    echo "Downloading $model..."
    model_name=$(echo $model | sed 's/\//_/g')
    huggingface-cli download "$model" --local-dir "$model_name"
done

# Download EAGLE v1/v2 models
# for model in "${models_eagle[@]}"; do
#     echo "Downloading $model..."
#     model_name=$(echo $model | sed 's/\//_/g')
#     huggingface-cli download "$model" --local-dir "$model_name"
# done

echo "All models downloaded successfully!"
