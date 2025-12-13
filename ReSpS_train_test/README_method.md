# ReSpS Methodology Implementation Guide

## Methodology Implementation Guide

This section provides a roadmap to locate the code implementations for the key methodological contributions described in the Re-SpS paper, specifically addressing the two main challenges and their solutions.

### MDP Formulation and Vanilla RL Solution

The core MDP formulation components are implemented across several key files:

- **State Space & Action Space**: `eagle/evaluation/optimized_sb3_discrete_ppo_online_rl_policy_ofl.py`
  - Lines 96-147: `OptimizedEagleParameterEnv` class defines the discrete action space with parameter bins
  - Lines 181-231: `_encode_state_from_hidden_states()` method implements state representation

- **Reward Function**: Multiple policy files implement Equation 3 (tokens per second reward)
  - `eagle/evaluation/optimized_sb3_discrete_ppo_online_rl_policy_ofl.py` (line 1800)
  - Function: `calculate_optimized_sb3_discrete_ppo_reward()`

- **PPO Implementation**: 
  - `eagle/evaluation/optimized_sb3_discrete_ppo_online_rl_policy_ofl.py`: Main optimized PPO policy
  - Uses Stable Baselines 3 PPO with max-entropy regularization (entropy coefficient tuning)

### Challenge 1: Expensive State Representation

**Problem**: SentenceBERT embedding computation (5-15ms per call) creates prohibitive overhead.

**Solution**: Efficient State Representation via Feature Reuse (Equation 4)

**Implementation Location**:
```
eagle/evaluation/optimized_sb3_discrete_ppo_online_rl_policy_ofl.py
```

**Key Methods**:
- **Lines 181-231**: `_encode_state_from_hidden_states()` 
  - Implements Equation 4: reuses EAGLE-3 internal features `[h^(h,m,l)_LM]`
  - Concatenates hidden states from three strategically selected LLM layers
  - **No additional inference cost** - reuses existing EAGLE-3 computations

**EAGLE-3 Integration**:
```
eagle/model/ea_model.py
```
- **Lines 274-290**: Shows how hidden states are extracted during EAGLE-3 generation
- **Lines 681-685**: Integration point where EAGLE-3 features are passed to RL policy

**Feature Processing**:
- **3k-dimensional concatenated features**: Direct concatenation from 3 layers
- **k-dimensional reduced features**: After internal FC layer processing  
- **Efficient handling**: Automatic dimension detection and processing

### Challenge 2: Frequent Policy Inference Overhead  

**Problem**: Policy network calls at every decoding step create cumulative computational bottleneck.

**Solution**: Multi-Step Action Persistence (Action Caching) - Algorithm 1

**Implementation Location**:
```
eagle/evaluation/optimized_sb3_discrete_ppo_online_rl_policy_ofl.py
```

**Key Methods**:
- **Lines 324-326**: Configuration parameters
  ```python
  action_cache_steps=10,        # Generate action every N steps (See Section 4.3 of Re-SpS paper)
  action_cache_enabled=True,    # Enable action caching
  ```

- **Lines 548-575**: Cache logic implementation (Algorithm 1)
  ```python
  # OPTIMIZATION 2: Action caching logic
  if (self.cached_params is not None and 
      self.cache_step_counter < self.action_cache_steps):
      # Use cached parameters (See Section 4.3 of Re-SpS paper)
  ```

- **Lines 805-820**: Reward aggregation over cache interval (Equation 5)
  ```python
  # Check if we should aggregate and update policy (See Algorithm 1 in Section 4.3 of Re-SpS paper)
  # Aggregate rewards over the cache period (see section 4.3 of Re-SpS paper)
  avg_reward = sum(self.cached_rewards) / len(self.cached_rewards)
  ```

### Re-SpS Complete Architecture

**Main Integration Point**:
```
eagle/model/ea_model.py
```
- **Lines 210-220**: RL policy integration in `eagenerate()` method
- **Lines 280-350**: Step-wise RL parameter prediction and caching
- **Lines 400-450**: Multi-step action persistence implementation

**Policy Selection**:
The paper's default setting and main results use the **Optimized SB3 Discrete PPO OFL** policy:
```
eagle/evaluation/optimized_sb3_discrete_ppo_online_rl_policy_ofl.py
```

### Key Configuration Parameters

**Training Configuration** (N=10):
- Cache interval during training: 10 steps
- Reward aggregation: Average over 10 consecutive steps
- Policy update frequency: Every 10 steps

**Inference Configuration** (N=30):  
- Cache interval during inference: 30 steps
- Reduces policy overhead by 97% (30x fewer calls)
- Maintains adaptivity while maximizing efficiency

### Usage Examples

**Run with Paper's Default Configuration**:
```bash
# Uses optimized_sb3_discrete_ppo_ofl policy with:
# - EAGLE-3 feature reuse (Challenge 1 solution)
# - Action caching N=30 for inference and N=10 for training (Challenge 2 solution)
bash generate_multi_gpu_scripts.sh
# Select option 5 for default paper configuration
```

**Manual Configuration**:
```bash
python -m eagle.evaluation.gen_ea_answer_llama3chat_rl \
  --use-online-rl \
  --use-optimized-sb3-discrete-ppo \
  --optimized-policy-version ofl \
  --use-eagle3-features \ # Reuse EAGLE-3 features
  --action-cache-steps 30 \
  --action-cache-enabled
```