# Official Code for Re-SpS: A Reinforcement Learning Approach to Speculative Sampling

This repository contains the official code for the paper "Re-SpS: A Reinforcement Learning Approach to Speculative Sampling" for statistical analysis of the RL policy.

This repository doesn't include training and evaluation code.

For training and evaluation code, refer to the `../ReSpS_train_test` directory. 

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
### Step 7: Analysis 

Assuming you have the log files from the training and evaluation in the `logs` directory, you can run the following analysis scripts.

This script will analyze the overhead tradeoff of the RL policy:
```bash
bash test_overhead_tradeoff.sh
```

These commands will analyze the statistical properties of the RL policy:
```bash
# Generate analysis for LLaMA 3.1-8B
python -m eagle.evaluation.comprehensive_statistical_analysis \
   --results-dir stats_test_data/results_llama3.1-8B/*_ofl128_llama318b \
   --tokenizer-path meta-llama/Llama-3.1-8B-Instruct
# Generate analysis for Vicuna 13B
python -m eagle.evaluation.comprehensive_statistical_analysis \
   --results-dir stats_test_data/results_vicuna-13b/*_ofl128_vicuna13b \
   --tokenizer-path lmsys/vicuna-13b-v1.5
or (the above is used in the paper):
python -m eagle.evaluation.comprehensive_statistical_analysis \
   --results-dir stats_test_data/results_vicuna-13b/*_ofl128_vicuna13b \
   --tokenizer-path lmsys/vicuna-13b-v1.3
# Generate analysis for LLaMA 3.3-70B
python -m eagle.evaluation.comprehensive_statistical_analysis \
   --results-dir stats_test_data/results_llama3.3_70b/*_ofl128_llama3370b \
   --tokenizer-path meta-llama/Llama-3.3-70B-Instruct
```

These commands will summarize the log files (speedup and unique actions) in a more readable format:
```bash
bash analyze_actions_clean.sh
python extract_summary_data_fixed.py
```
Make sure to place the above two scripts in the directory where it has `log` directory as a subdirectory.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.