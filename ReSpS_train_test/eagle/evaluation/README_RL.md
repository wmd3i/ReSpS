# EAGLE RL Parameter Optimization

This directory contains the implementation for using Reinforcement Learning (RL) to optimize EAGLE tree generation parameters (`total_tokens`, `depth`, `top_k`) dynamically based on input context.

## Components

### 1. RL Environment (`rl_tree_policy.py`)
- `TreeRLEnvironment`: Gym environment for training RL policies
- `RLTreePolicy`: Wrapper for trained policies to predict parameters
- Reward function based on generation speed, acceptance rate, and efficiency

### 2. Modified Evaluation Script (`gen_ea_answer_llama3chat_rl.py`)
- Enhanced version that supports RL policy-based parameter prediction
- Collects training data for RL optimization
- Falls back to command-line arguments when RL policy is not available

### 3. Training Script (`train_rl_policy.py`)
- Trains PPO policies using collected data or simulated environments
- Supports both initial training and refinement from real data

## Setup

1. Install additional dependencies:
```bash
pip install -r requirements-rl.txt
```

2. Make sure you have the base EAGLE environment set up according to the main README.

## Usage

### Option 1: Train RL Policy from Questions (Initial Training)

```bash
# Train initial policy using MT-Bench questions
cd eagle/evaluation
python train_rl_policy.py \
    --mode from_questions \
    --question-file ../data/mt_bench/question.jsonl \
    --policy-path ppo_tree_policy_initial.zip \
    --total-timesteps 1000
```

### Option 2: Collect Real Performance Data

```bash
# Run evaluation with data collection enabled
python gen_ea_answer_llama3chat_rl.py \
    --base-model-path /path/to/base/model \
    --ea-model-path /path/to/eagle/model \
    --model-id llama3-rl-test \
    --collect-rl-data \
    --rl-data-file rl_training_data.jsonl \
    --total-token 60 \
    --depth 5 \
    --top-k 10
```

### Option 3: Train RL Policy from Collected Data

```bash
# Refine policy using real performance data
python train_rl_policy.py \
    --mode from_data \
    --data-file rl_training_data.jsonl \
    --policy-path ppo_tree_policy_refined.zip \
    --total-timesteps 2000
```

### Option 4: Use Trained RL Policy for Evaluation

```bash
# Run evaluation with RL policy (no need to specify tree parameters)
python gen_ea_answer_llama3chat_rl.py \
    --base-model-path /path/to/base/model \
    --ea-model-path /path/to/eagle/model \
    --model-id llama3-rl-optimized \
    --use-rl-policy \
    --rl-policy-path ppo_tree_policy_refined.zip

# Or use the convenience script
./run_rl_evaluation.sh
```

**Note**: When `--use-rl-policy` is enabled, the traditional parameters (`--total-token`, `--depth`, `--top-k`) serve as fallback values and for initial model setup. The RL policy will dynamically predict optimal parameters for each input.

## RL Components Explanation

### State Representation
- **Input**: Current conversation context (prompt + previous turns)
- **Encoding**: SBERT embedding (384-dimensional vector)
- **Purpose**: Captures semantic meaning to guide parameter selection

### Action Space
- **total_tokens**: [32, 48, 64, 80, 96] (discrete choices)
- **depth**: [2, 4, 6, 8, 10] (discrete choices)  
- **top_k**: [4, 8, 12, 16, 20] (discrete choices)
- **Type**: MultiDiscrete action space

### Reward Function
```python
def calculate_reward(generation_time, acceptance_rate, num_accepted_tokens, 
                     total_tokens, depth, top_k):
    speed = num_accepted_tokens / generation_time
    speed_reward = np.log(speed + 1) * 2.0
    acceptance_reward = acceptance_rate * 3.0
    efficiency_penalty = (total_tokens + depth + top_k) * 0.01
    quality_bonus = 2.0 if acceptance_rate > 0.8 and speed > 10 else 0.0
    return speed_reward + acceptance_reward - efficiency_penalty + quality_bonus
```

### Policy Network
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network**: MLP (Multi-Layer Perceptron)
- **Input**: 384-dim SBERT embedding
- **Output**: Discrete action probabilities for each parameter

## Advanced Usage

### Custom Parameter Ranges
To modify the parameter ranges, edit the bins in `rl_tree_policy.py`:

```python
self.total_token_bins = [32, 48, 64, 80, 96]  # Modify as needed
self.depth_bins = [2, 4, 6, 8, 10]           # Modify as needed
self.top_k_bins = [4, 8, 12, 16, 20]         # Modify as needed
```

### Custom Reward Function
Modify the `calculate_reward` function in `rl_tree_policy.py` to emphasize different aspects:

```python
def calculate_reward(generation_time, acceptance_rate, num_accepted_tokens, 
                     total_tokens, depth, top_k):
    # Emphasize speed more
    speed_reward = np.log(speed + 1) * 5.0  # Increased weight
    
    # Add custom metrics
    efficiency_ratio = num_accepted_tokens / total_tokens
    custom_bonus = 1.0 if efficiency_ratio > 0.9 else 0.0
    
    return speed_reward + acceptance_reward - efficiency_penalty + custom_bonus
```

### Monitoring Training
Use TensorBoard to monitor training progress:

```bash
# Start tensorboard (in a separate terminal)
tensorboard --logdir ./tensorboard_logs

# Train with logging enabled
python train_rl_policy.py --mode from_questions --total-timesteps 5000
```

## Files Overview

- `rl_tree_policy.py`: Core RL components and environment
- `gen_ea_answer_llama3chat_rl.py`: Modified evaluation script with RL support  
- `train_rl_policy.py`: Policy training script
- `requirements-rl.txt`: Additional Python dependencies
- `README_RL.md`: This documentation file

## Troubleshooting

### Common Issues

1. **ImportError for stable_baselines3**: Install RL requirements
2. **CUDA out of memory**: Reduce batch size or model parameters
3. **Policy not improving**: Collect more diverse training data or adjust reward function
4. **Slow training**: Reduce total_timesteps or use smaller model

### Performance Tips

1. **Data Collection**: Run evaluation on diverse question sets for better policy generalization
2. **Training**: Start with simulated training, then refine with real data
3. **Evaluation**: Use deterministic policy prediction for consistent results
4. **Monitoring**: Track reward trends to ensure policy is learning effectively
