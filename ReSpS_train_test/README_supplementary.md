# 📚 Other General Guidelines (Optional, can be skipped and not for the default configuration in the paper)

This guide covers the usage of two powerful scripts for EAGLE RL training and evaluation:

1. `generate_multi_gpu_scripts.sh` - Multi-GPU script generator, using the `test_optimized_ppo_modes_comparison_single.sh` as a template. (Default way in the paper, recommended)
As the default configuration in the paper shown above, you can run all the configurations in the paper and ablation study using this script. If you decide to use this script, you can skip the `test_optimized_ppo_modes_comparison_single.sh` script.

2. `test_optimized_ppo_modes_comparison_single.sh` (Optional) - Single-GPU comprehensive training script. This script is used to run the all the configurations in one machine, usually for debugging purposes.

---

## 🔧 `generate_multi_gpu_scripts.sh`

### Overview

This script generates multiple GPU-specific training scripts based on your configuration preferences using the `test_optimized_ppo_modes_comparison_single.sh` as a template. 
It automates the creation of scripts that can run different parameter combinations across multiple GPUs simultaneously.

### Key Features

- **Multi-GPU Distribution**: Automatic round-robin or custom GPU assignment
- **Pre-built Combinations**: 8 standard parameter combinations
- **Custom Parameters**: Define your own parameter combinations
- **Model Selection**: Support for LLaMA3.1-8B, Vicuna-13B, LLaMA3.3-70B
- **Network Architecture**: Configurable for each policy version
- **Date/Time Management**: Flexible naming and organization

### Usage Examples

#### 1. Interactive Mode (Recommended)
```bash
bash generate_multi_gpu_scripts.sh
```

This will guide you through:
- Model selection
- Combination mode (pre-built vs custom)
- GPU assignment method
- Network architecture configuration
- Output directory naming

#### 2. Non-Interactive Mode - All Pre-built Combinations
```bash
# Generate all 8 pre-built combinations
bash generate_multi_gpu_scripts.sh 1 2 3 4 5 6 7 8
```

#### 3. Specific Pre-built Combinations
```bash
# Generate only combinations 1, 3, 5, 7
bash generate_multi_gpu_scripts.sh 1 3 5 7
```

#### 4. With Custom Date/Time
```bash
# Use custom timestamp
bash generate_multi_gpu_scripts.sh --datetime 20250804_1200 1 2 3 4
```

#### 5. Overwrite Existing Scripts
```bash
# Reuse existing folder structure
bash generate_multi_gpu_scripts.sh --overwrite 1 2 3 4
```

### Pre-built Combinations

The script offers 8 standard combinations:

1. **Standard Version + Standard State + Max Entropy**
2. **Standard Version + Standard State + No Max Entropy** 
3. **Standard Version + Context Only + Max Entropy**
4. **Standard Version + Context Only + No Max Entropy**
5. **OFL Version + Standard State + Max Entropy**
6. **OFL Version + Standard State + No Max Entropy**
7. **OFL Version + Context Only + Max Entropy** 
8. **OFL Version + Context Only + No Max Entropy**

### Custom Parameters Mode

Define your own combinations using 6 binary flags:

```bash
# Format: "RUN_STANDARD_VERSION RUN_OFL_VERSION RUN_STANDARD RUN_CONTEXT_ONLY RUN_MAX_ENTROPY RUN_NO_MAX_ENTROPY"
# Example: "1 0 1 0 1 0" = Standard Version + Standard State + Max Entropy
```

### Generated Output

The script creates:

```
test_op__llama318b_multi_gpu_20250804_1200/
├── test_op__gpu0_combo1_stdver_stdstate_maxent.sh
├── test_op__gpu1_combo2_stdver_stdstate_noent.sh
├── test_op__gpu2_combo3_stdver_context_maxent.sh
├── test_op__gpu3_combo4_stdver_context_noent.sh
├── run_all_scripts.sh          # Master script to run all
├── launch.sh                   # Simple launcher
└── script_summary.txt          # Configuration summary
```

### Running Generated Scripts

#### Option 1: Run All Scripts
```bash
cd test_op__llama318b_multi_gpu_20250804_1200/
bash run_all_scripts.sh
```

#### Option 2: Use Launcher
```bash
cd test_op__llama318b_multi_gpu_20250804_1200/
bash launch.sh
```

#### Option 3: Run Individual Scripts
```bash
cd test_op__llama318b_multi_gpu_20250804_1200/
bash test_op__gpu0_combo1_stdver_stdstate_maxent.sh
```

### GPU Assignment Methods

#### 1. Round-Robin (Automatic)
- Scripts distributed evenly across available GPUs
- Good for balanced workloads

#### 2. Custom Assignment
```bash
# Example assignments:
# "0,1;2,3;4,5;6,7" - 4 scripts, each using 2 GPUs
# "0;1;2;3" - 4 scripts, each using 1 GPU  
# "-1;0;1;2" - First script CPU-only, others on GPUs
```

---

## 📄 `test_optimized_ppo_modes_comparison_single.sh` (Optional)

### Overview

This is a comprehensive training and evaluation script that supports multiple policy modes, benchmarks, and configurations. It's designed to run various combinations of EAGLE RL optimizations on a single GPU.

### Key Features

- **Multiple Policy Versions**: Standard and OFL (Optimized Feature Learning) versions
- **State Representation Modes**: Standard vs Context-only state representation
- **Entropy Modes**: Max-entropy vs Standard PPO configurations
- **Multiple Benchmarks**: MT-Bench, HumanEval, GSM8K, Alpaca, CNN/DailyMail, Natural Questions
- **Configurable Network Architecture**: Customizable neural network layers

### Configuration Options

#### Model Configuration
```bash
# Models supported (automatically configured by script generator)
MODEL_NAME="LLaMA3.1-8B"              # or "Vicuna-13B", "LLaMA3.3-70B"
MODEL_PATH="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
```

#### Execution Mode Control
```bash
# Policy version selection
RUN_STANDARD_VERSION=1   # Run standard policy version (0=disable, 1=enable)
RUN_OFL_VERSION=1       # Run OFL policy version with enhanced features

# State representation modes
RUN_STANDARD=1          # Run without --use-context-only-state  
RUN_CONTEXT_ONLY=1      # Run with --use-context-only-state

# Entropy modes  
RUN_MAX_ENTROPY=1       # Run with max-entropy PPO
RUN_NO_MAX_ENTROPY=1    # Run without max-entropy (standard PPO)
```

#### Network Architecture
```bash
# Standard version: comma-separated integers
STANDARD_NET_ARCH="128,128"  # Example: "512,256,128"

# OFL version: same format or different pi/vf networks
OFL_NET_ARCH="128,128"      # Example: "64,64;128,128" for different pi/vf
```

### Usage Examples

#### 1. Run All Configurations for LLaMA3.1-8B
```bash
# Run the script as-is - will train all enabled combinations
bash test_optimized_ppo_modes_comparison_single.sh
```
For Vicuna-13B or LLaMA3.3-70B, edit the `MODEL_NAME` and `MODEL_PATH` variables in the script.

#### 2. Run Only OFL Version with Max Entropy (Default Configuration in the paper)
```bash
# Edit the script to set:
RUN_STANDARD_VERSION=0
RUN_OFL_VERSION=1
RUN_STANDARD=1
RUN_CONTEXT_ONLY=0
RUN_MAX_ENTROPY=1
RUN_NO_MAX_ENTROPY=0

bash test_optimized_ppo_modes_comparison_single.sh
```

#### 3. Run OFL Version with Context-Only State and No Max Entropy
(One of the configurations in ablation study)
```bash
# Edit the script to set:
RUN_STANDARD_VERSION=0
RUN_OFL_VERSION=1
RUN_STANDARD=0
RUN_CONTEXT_ONLY=1
RUN_MAX_ENTROPY=0
RUN_NO_MAX_ENTROPY=1

bash test_optimized_ppo_modes_comparison_single.sh
```

#### 4. Custom Network Architecture
You can customize the network architecture (default configuration in the paper):

```bash
# For deeper networks, edit in script:
STANDARD_NET_ARCH="128,128"
OFL_NET_ARCH="128,128"  # Different architectures for policy/value
```

### Output Structure

The script creates organized output directories:

```
log/YYYYMMDD_HHMM_optimized_ppo/
├── optimized_max_entropy_ppo_standard_ofl/
│   ├── checkpoints/
│   ├── evaluation/
│   ├── baseline_results/
│   ├── training.log
│   ├── summary.txt
│   └── execution_config.txt
├── optimized_standard_ppo_context_ofl/
│   └── ... (same structure)
└── ... (other enabled configurations)
```
The `optimized_max_entropy_ppo_standard_ofl` is the default configuration used in the paper.

---

## 📊 Monitoring and Results

### With Weights & Biases

```bash
# Enable wandb logging
python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
    --use-online-rl \
    --use-wandb \
    --wandb-project "eagle-rl-optimization" \
    --wandb-run-name "test-run-001" \
    # ... other arguments
```

### Log Files

- **Training logs**: `training.log` in each policy directory
- **Evaluation logs**: `evaluation/*.log`

### Key Metrics to Monitor

- **Tokens per second**: Generation speed
- **Unique actions**: Diversity of actions taken

### Result Analysis

```bash
# View training summary
cat log/*/optimized_*/summary.txt

# Check final performance
grep "Final" log/*/optimized_*/training.log

# Compare configurations
cat log/*/optimized_*/comparison.txt
```

---

## 🛠️ Troubleshooting

### Common Issues

4. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x *.sh
   ```

---

## 📝 Summary

- **Single GPU**: Use `test_optimized_ppo_modes_comparison_single.sh` directly
- **Multi-GPU**: Use `generate_multi_gpu_scripts.sh` to create distributed scripts
- **Customization**: Edit configuration variables for specific needs
- **Monitoring**: Check log files for progress and results

---


## 🎯 RL Policy Types

### Available Policy Types:

1. **`optimized_sb3_discrete_ppo_ofl`** - Optimized SB3 PPO with layer features (default in the paper)
2. **`optimized_sb3_discrete_ppo`** - Standard optimized SB3 PPO
3. **`sb3_discrete_ppo`** - Basic SB3 discrete PPO  (deprecated)
4. **`discrete_ppo`** - Custom discrete PPO implementation (deprecated)
5. **`optimized_online`** - Optimized online policy  (deprecated)

### Key Features:

- **Action Caching**: Reduces computation by reusing actions for N steps
- **EAGLE-3 Features**: Uses layer concatenation instead of SBERT embeddings
- **Context-Only State**: Optional SBERT-only state representation
- **Dynamic Parameters**: Automatically adjusts `total_tokens`, `depth`, and `top_k`

## 🔧 Configuration Options

### Command Line Arguments

```bash
# Core EAGLE parameters
--total-token 64          # Number of draft tokens
--depth 5                 # Tree depth
--top-k 10               # Top-k sampling

# RL-specific parameters
--online-lr 0.001         # Learning rate
--action-cache-steps 10   # Action caching frequency
--online-repeat-factor 3  # Training data repetition

# Performance options
--use-context-only-state  # Use SBERT embeddings only
--use-eagle3-features    # Use EAGLE-3 layer features
--max-entropy-inference  # Enable entropy-based sampling
```

## 📈 Performance Monitoring



## 🏗️ Project Structure

```
./
├── eagle/
│   ├── evaluation/
│   │   ├── optimized_sb3_discrete_ppo_online_rl_policy_ofl.py
│   │   ├── gen_ea_answer_llama3chat_rl.py
│   │   └── gen_ea_answer_vicuna_rl.py
│   ├── model/
│   │   └── ea_model.py
│   └── traineagle3/
├── download_dataset_scripts/
│   ├── setup_conversational_training.sh
│   └── convert_*.py
├── eagle-rl-h100.yml      # H100 environment
├── eagle-rl-a40.yml       # A40/RTX environment
├── requirements.txt       # Basic requirements
├── requirements-rl.txt    # RL requirements
└── README.md          # Documentation
```

## 🔍 Troubleshooting

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt -r requirements-rl.txt
   ```

2. **Model Download Issues**
   ```bash
   # Check Hugging Face authentication
   huggingface-cli whoami
   ```
