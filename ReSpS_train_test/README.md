# Official Code for Re-SpS: A Reinforcement Learning Approach to Speculative Sampling

This repository contains the official code for the paper "Re-SpS: A Reinforcement Learning Approach to Speculative Sampling" for training and evaluating the RL policy. 

This repository doesn't include statistic analysis.

For statistical analysis of the RL policy, refer to the `../ReSps_stats` directory.

## 📋 Prerequisites

- Linux environment
- CUDA-compatible GPU (RTX 3090/4090, A40, H100, etc.)
- Python 3.11+
- Git
- Conda/Miniconda

## 🛠️ Installation

### Create Conda Environment

Choose the appropriate environment file based on your GPU:

**For H100 GPUs (for Llama-3.3-70B):**
```bash
conda env create -f eagle-rl-h100.yml
conda activate eagle-rl
```

**For A40/RTX GPUs (for Llama-3.1-8B & Vicuna-13B):**
```bash
conda env create -f eagle-rl-a40.yml
conda activate eagle-rl
```

### Step 4: Install PyTorch (if needed)

```bash
# For CUDA 12.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Step 5: Install Additional Dependencies

```bash
# Basic requirements
pip install -r requirements.txt

# RL-specific requirements
pip install -r requirements-rl.txt

# Additional packages for data processing
pip install huggingface-hub pandas
```

### Step 6: Hugging Face Authentication

```bash
huggingface-cli login
```

## 📊 Dataset Setup

### Download and Process Training Data

```bash
# Setup conversational training data
bash download_dataset_scripts/setup_conversational_training.sh
bash download_dataset_scripts/setup_rl_with_combined_dataset.sh
```

This will:
- Download ShareGPT and UltraChat-200K datasets
- Validate and combine datasets
- Convert them to correct format

## 🏃‍♂️ Run Default Configuration in the Paper

```bash
bash generate_multi_gpu_scripts.sh

# Below is the default configuration selection in the paper:

# Model selection: Choose one from LLaMA3.1-8B, Vicuna-13B, LLaMA3.3-70B

# Model Name Inclusion: Use default

# Enter number of available GPUs: 1 for single A40 NVIDIA GPU (Llma3.1-8B or Vicuna-13B) or 4 for four H100 NVIDIA GPU (LLaMA3.3-70B)

# Combination Mode Selection: Use default

# Pre-built Combinations: 5 for the default configuration in the paper, "5 6 7 8" for the configurations in ablation study

# Enter network architecture for OFL version: "128,128" for the default configuration in the paper. "64,64" for the configurations in ablation study.

# Choose GPU assignment method: 1 (use round-robin assignment) for single A40 NVIDIA GPU (Llma3.1-8B or Vicuna-13B) or 2 (custom assignment) for four H100 NVIDIA GPU (LLaMA3.3-70B)

# (Optional) If selected 2 for the above step, enter custom GPU assignment: "0,1,2,3;0,1,2,3;0,1,2,3;0,1,2,3" for four H100 NVIDIA GPU (LLaMA3.3-70B)

# Date/Time options: Use default (this will create a folder with current date and time)

# Do you want to run all generated scripts now: Use default (no)
```
For more general usage, refer to the `README_supplementary.md` file.

## 🧩 Methodology Implementation Guide

For the Re-SpS paper, we present a comprehensive guide to the implementation of the methodology described in the paper. This guide focuses on the two main challenges addressed in the paper and provides detailed information on how to locate and understand the code implementations. Please refer to the `README_method.md` file for the guide.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.