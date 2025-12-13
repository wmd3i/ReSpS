"""
Optimized DQN Online RL Policy for Real-time EAGLE Parameter Optimization
Optimizations:
1. Layer Feature Concatenation (EAGLE-3 features instead of SBERT)
2. Action Generation Frequency optimization (caching actions for N steps)
"""

import numpy as np
import torch
import torch.nn as nn
import os
import json
from collections import deque
import random

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

class OptimizedOnlineTreePolicy:
    """Optimized Max-Entropy Online RL Policy that learns tree parameters with EAGLE-3 features and action caching
    
    Optimizations:
    1. Uses EAGLE-3 layer features instead of SBERT text embeddings
    2. Action caching to reduce computation frequency (generate action every N steps)
    3. NEW: Context-only state representation (SBERT embeddings without projection)
    """
    
    def __init__(self, 
                 learning_rate=3e-4,
                 epsilon_start=0.9,
                 epsilon_end=0.05,
                 epsilon_decay=0.995,
                 memory_size=1000,
                 batch_size=32,
                 target_update_freq=100,
                 # Max-entropy RL parameters
                 temperature=1.0,
                 entropy_weight=0.1,
                 inference_temperature=1.5,
                 max_entropy_inference=True,
                 # OPTIMIZATION 2: Action caching parameters
                 action_cache_steps=10,        # Generate action every N steps
                 action_cache_enabled=True,    # Enable action caching
                 # OPTIMIZATION 1: EAGLE-3 feature parameters
                 hidden_size=4096,             # Model hidden size (k)
                 use_eagle3_features=True,     # Use EAGLE-3 features instead of SBERT
                 # NEW: Context-only state representation option
                 use_context_only_state=False, # Use SBERT context embeddings directly (384D)
                 use_wandb=True,
                 wandb_project="eagle-optimized-online-rl",
                 wandb_run_name=None,
                 checkpoint_dir="optimized_dqn_checkpoints",
                 checkpoint_freq=100,
                 max_checkpoints=3):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # NEW: Context-only state representation configuration
        self.use_context_only_state = use_context_only_state
        
        # Validate configuration
        if use_context_only_state and use_eagle3_features:
            print("⚠️  Warning: use_context_only_state=True overrides use_eagle3_features. Using context-only state.")
            use_eagle3_features = False
        
        # OPTIMIZATION 1: EAGLE-3 features configuration
        self.use_eagle3_features = use_eagle3_features
        self.hidden_size = hidden_size if not use_context_only_state else 384
        self.feature_dim = hidden_size if not use_context_only_state else 384  # Use 384 for context-only mode
        
        # OPTIMIZATION 2: Action caching configuration
        self.action_cache_enabled = action_cache_enabled
        self.action_cache_steps = action_cache_steps
        self.cached_action = None
        self.cached_params = None
        self.cache_step_counter = 0
        self.cache_hidden_states = None
        
        # NEW: Reward aggregation for cached actions
        self.cached_rewards = []
        self.cached_generation_times = []
        self.cached_new_tokens = []
        self.last_cache_update_step = 0
        
        # Max-entropy RL configuration
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.inference_temperature = inference_temperature
        self.max_entropy_inference = max_entropy_inference
        
        # Resume mechanism configuration
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        self.questions_processed = 0
        self.training_seed = None
        # NEW: Track last checkpoint step to prevent multiple saves
        self.last_checkpoint_step = -1
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize wandb logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb_run_name = wandb_run_name or f"optimized-eagle-dqn-{int(torch.randint(0, 10000, (1,)).item())}"
            wandb.init(project=wandb_project, name=wandb_run_name, reinit=True)
            wandb.config.update({
                "learning_rate": learning_rate,
                "epsilon_start": epsilon_start,
                "epsilon_end": epsilon_end,
                "epsilon_decay": epsilon_decay,
                "temperature": temperature,
                "entropy_weight": entropy_weight,
                "action_cache_steps": action_cache_steps,
                "action_cache_enabled": action_cache_enabled,
                "use_eagle3_features": use_eagle3_features,
                "use_context_only_state": use_context_only_state,
                "hidden_size": self.hidden_size,
                "feature_dim": self.feature_dim
            })
        else:
            print("Warning: Wandb logging disabled")
        
        # Expanded parameter bins for more granularity - best combination ranges
        self.top_k_bins = [8, 12, 16, 20, 32]
        self.depth_bins = [3, 4, 5, 6, 7, 8]
        self.total_tokens_bins = [32, 48, 64, 80, 96, 128]
        
        # Action space dimensions
        self.n_total_tokens = len(self.total_tokens_bins)
        self.n_depth = len(self.depth_bins)
        self.n_top_k = len(self.top_k_bins)
        self.total_actions = self.n_total_tokens * self.n_depth * self.n_top_k
        
        # Precompute valid actions to avoid runtime constraint violations
        self.valid_actions = self._precompute_valid_actions()
        print(f"Action space: {self.total_actions} total, {len(self.valid_actions)} valid")
        
        # DQN parameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks using EAGLE-3 features
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        self.step_count = 0
        
        # Training state
        self.last_state = None
        self.last_action = None
        self.last_params = None
        self.last_exploration_mode = None
        self.last_entropy = 0.0
        
        # Performance tracking
        self.reward_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        self.entropy_history = deque(maxlen=1000)
        
        print(f"OptimizedOnlineTreePolicy initialized:")
        print(f"  - Valid actions: {len(self.valid_actions)}/{self.total_actions}")
        print(f"  - OPTIMIZATION 1 - EAGLE-3 features: {'enabled' if use_eagle3_features else 'disabled'}")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - OPTIMIZATION 2 - Action caching: {'enabled' if action_cache_enabled else 'disabled'}")
        if action_cache_enabled:
            print(f"    - Cache steps: {action_cache_steps} (generate action every {action_cache_steps} steps)")
        print(f"  - Max-entropy inference: {'enabled' if max_entropy_inference else 'disabled'}")
        if max_entropy_inference:
            print(f"    - Inference temperature: {inference_temperature}")
        print(f"  - Device: {self.device}")
        print(f"  - Wandb logging: {'enabled' if self.use_wandb else 'disabled'}")

    # def _build_network(self):
    #     """Build Q-network for EAGLE-3 features"""
    #     network = nn.Sequential(
    #         nn.Linear(self.feature_dim, 64),
    #         nn.ReLU(),
    #         nn.Dropout(0.2),
    #         nn.Linear(64, 64),
    #         nn.ReLU(),
    #         nn.Dropout(0.2),
    #         nn.Linear(64, len(self.valid_actions))  # Output for valid actions only
    #     )
        
    #     # Initialize weights
    #     for layer in network:
    #         if isinstance(layer, nn.Linear):
    #             nn.init.xavier_uniform_(layer.weight)
    #             nn.init.constant_(layer.bias, 0.1)
        
    #     return network    
    def _build_network(self):
        """Build Q-network for EAGLE-3 features"""
        network = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.valid_actions))  # Output for valid actions only
        )
        
        # Initialize weights
        for layer in network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)
        
        return network
    
    def _encode_state_from_hidden_states(self, hidden_states):
        """OPTIMIZATION 1: Encode state from EAGLE-3 layer features instead of SBERT
        
        Args:
            hidden_states: Concatenated hidden states from EAGLE-3 (3k dimensions)
                          or already reduced features (k dimensions)
        
        Returns:
            torch.Tensor of shape (feature_dim,) ready for RL policy
        """
        if hidden_states is None:
            # Return zero state if no hidden states available
            return torch.zeros(self.feature_dim, device=self.device, dtype=torch.float32)
        
        # Convert to tensor and ensure correct device and dtype
        if isinstance(hidden_states, np.ndarray):
            features = torch.FloatTensor(hidden_states).to(self.device)
        elif isinstance(hidden_states, torch.Tensor):
            features = hidden_states.float().to(self.device)  # Ensure float32 dtype
        else:
            features = torch.tensor(hidden_states, device=self.device, dtype=torch.float32)
        
        # Handle different input shapes
        if len(features.shape) > 1:
            # If we have (batch, sequence, features), take the last sequence position
            if len(features.shape) == 3:  # (batch, sequence, features)
                features = features[0, -1]  # Take last sequence position from first batch
            elif features.shape[0] == 1:  # (1, features) - remove batch dimension
                features = features[0]
            else:  # (sequence, features) - take last sequence position
                features = features[-1]
        
        # Handle different feature dimensions
        if features.shape[-1] == self.hidden_size * 3:
            # This is the 3k-dimensional concatenated features
            # In practice, this should be reduced by the FC layer in EAGLE-3
            # For now, we'll take the mean across the 3 feature groups
            features = features.view(3, self.hidden_size).mean(dim=0)
        elif features.shape[-1] == self.hidden_size:
            # This is already the k-dimensional reduced features
            pass
        else:
            # Unexpected dimension, pad or truncate to match expected size
            if features.shape[-1] > self.hidden_size:
                features = features[:self.hidden_size]
            else:
                padded = torch.zeros(self.hidden_size, device=self.device, dtype=torch.float32)
                padded[:features.shape[-1]] = features
                features = padded
        
        # Ensure final tensor is float32 for network compatibility
        return features.float()
    
    @torch.no_grad()
    def _encode_state_from_context(self, context):
        """Encode conversation context using SBERT embeddings
        
        NEW: Supports both context-only mode (384D) and legacy EAGLE-3 mapping mode (4096D)
        """
        if not hasattr(self, '_sbert_model'):
            from sentence_transformers import SentenceTransformer
            self._sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        with torch.no_grad():
            sbert_embedding = self._sbert_model.encode(context)  # Shape: (384,)
        
        # NEW: Context-only state mode - use SBERT embeddings directly
        if self.use_context_only_state:
            # Return SBERT embeddings directly without projection
            sbert_tensor = torch.FloatTensor(sbert_embedding).to(self.device)
            
            if not hasattr(self, '_context_only_warned'):
                print(f"✅ Using context-only state representation:")
                print(f"   Input: Context text → SBERT embedding: {sbert_embedding.shape}")
                print(f"   Output: Direct SBERT features (384D) - no projection")
                print(f"   Feature density: 100% (native SBERT)")
                self._context_only_warned = True
            
            return sbert_tensor
        
        # Legacy mode: Map SBERT features to EAGLE-3 space
        # SOLUTION: Intelligent feature mapping instead of zero-padding
        # Use learned projection to maintain feature density and avoid 90% padding
        if not hasattr(self, '_feature_mapper'):
            # Create a learnable mapping from SBERT (384) to EAGLE-3 space (4096)
            self._feature_mapper = torch.nn.Sequential(
                torch.nn.Linear(384, self.feature_dim),
                torch.nn.LayerNorm(self.feature_dim),  # Normalize to match EAGLE-3 scale
                torch.nn.Tanh()  # Keep values in reasonable range
            ).to(self.device)
            
            # Initialize with small weights to start conservative
            with torch.no_grad():
                for module in self._feature_mapper:
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                        torch.nn.init.zeros_(module.bias)

            print(f"🔧 Created SBERT→EAGLE-3 feature mapper for DQN policy (384→{self.feature_dim})")
            print("   This avoids sparse padding and maintains feature density")
        
        # Map SBERT features to EAGLE-3 compatible space
        sbert_tensor = torch.FloatTensor(sbert_embedding).unsqueeze(0).to(self.device)
        mapped_features = self._feature_mapper(sbert_tensor).squeeze(0)
        
        # Log the mapping for monitoring
        if not hasattr(self, '_mapping_warned'):
            print(f"⚠️  Using SBERT fallback with learned mapping:")
            print(f"   Input: SBERT {sbert_embedding.shape} → Output: {mapped_features.shape}")
            print(f"   Feature density: 100% (no zero padding)")
            print(f"   Recommendation: Use EAGLE-3 features for best performance")
            self._mapping_warned = True
        
        return mapped_features
    
    def _action_to_params(self, action):
        """Convert discrete action to parameter values"""
        total_tokens_idx = action // (self.n_depth * self.n_top_k)
        remaining = action % (self.n_depth * self.n_top_k)
        depth_idx = remaining // self.n_top_k
        top_k_idx = remaining % self.n_top_k
        
        return (
            self.total_tokens_bins[total_tokens_idx],
            self.depth_bins[depth_idx],
            self.top_k_bins[top_k_idx]
        )
    
    def _params_to_action(self, total_tokens, depth, top_k):
        """Convert parameters to discrete action"""
        try:
            total_tokens_idx = self.total_tokens_bins.index(total_tokens)
            depth_idx = self.depth_bins.index(depth)
            top_k_idx = self.top_k_bins.index(top_k)
            return total_tokens_idx * (self.n_depth * self.n_top_k) + depth_idx * self.n_top_k + top_k_idx
        except ValueError:
            # If parameters not in bins, return random action
            return random.randint(0, self.total_actions - 1)
    
    def predict_parameters(self, context=None, hidden_states=None, training_mode=True):
        """OPTIMIZED: Predict parameters using EAGLE-3 features with action caching
        
        NEW: Supports context-only state representation (384D) and EAGLE-3 features (4096D)
        """
        
        # OPTIMIZATION 2: Action caching logic
        if self.action_cache_enabled:
            # Check if we can use cached action
            if (self.cached_params is not None and 
                self.cache_step_counter < self.action_cache_steps):
                
                self.cache_step_counter += 1
                
                # Log cache hit
                if training_mode and self.use_wandb and self.cache_step_counter % 5 == 0:
                    wandb.log({
                        "cache_hit": 1,
                        "cache_step": self.cache_step_counter,
                        "cached_params": {
                            "total_tokens": self.cached_params[0],
                            "depth": self.cached_params[1], 
                            "top_k": self.cached_params[2]
                        }
                    })
                
                # if self.cache_step_counter % 5 == 0:  # Log every 5th cache hit to avoid spam
                #     print(f"  Cache hit: Step {self.cache_step_counter}/{self.action_cache_steps} - Using cached params: {self.cached_params}")
                
                return self.cached_params
        
        # NEW: Context-only state mode
        if self.use_context_only_state:
            if context is None:
                return 96, 8, 20  # Default parameters
            
            # OPTIMIZATION 2: Action caching logic (works with context-only state too)
            if self.action_cache_enabled:
                if (self.cached_params is not None and 
                    self.cache_step_counter < self.action_cache_steps):
                    
                    self.cache_step_counter += 1
                    # if self.cache_step_counter % 5 == 0:  # Log every 5th cache hit to avoid spam
                    #     print(f"  Cache hit: Step {self.cache_step_counter}/{self.action_cache_steps} - Using cached params: {self.cached_params}")
                    return self.cached_params
            
            # Use context directly for state representation (384D)
            state = self._encode_state_from_context(context)
            feature_source = "CONTEXT_ONLY"
            
            # Track usage for monitoring
            if training_mode and self.use_wandb:
                wandb.log({
                    "feature_source": "context_only",
                    "context_length": len(context),
                    "state_dimension": 384,
                    "use_context_only_state": True
                })
        
        # Original EAGLE-3 features mode
        elif self.use_eagle3_features and hidden_states is not None:
            # Encode state from EAGLE-3 layer features
            state = self._encode_state_from_hidden_states(hidden_states)
            feature_source = "EAGLE3"
            
            # Track usage for monitoring
            if training_mode and self.use_wandb:
                wandb.log({
                    "feature_source": "eagle3",
                    "sbert_usage": 0,
                    "eagle3_usage": 1,
                    "use_context_only_state": False
                })
        elif self.use_eagle3_features and hidden_states is None:
            # EAGLE-3 mode but no hidden states available
            return 96, 8, 20  # Default parameters
        else:
            # Fallback to context-based encoding (SBERT→EAGLE-3 mapping)
            if context is not None:
                state = self._encode_state_from_context(context)
                feature_source = "SBERT_MAPPED"
                
                # Track usage for monitoring
                if training_mode and self.use_wandb:
                    wandb.log({
                        "feature_source": "sbert_mapped",
                        "sbert_usage": 1,
                        "eagle3_usage": 0,
                        "use_context_only_state": False
                    })
            else:
                # No input available (neither context nor hidden_states)
                return 96, 8, 20  # Default parameters
        
        # Max-entropy action selection with temperature-based sampling
        if self.max_entropy_inference and not training_mode:
            # Inference mode with max-entropy: use temperature-based sampling for diversity
            action, entropy = self._sample_action_with_temperature(state, self.inference_temperature)
            exploration_mode = "MAX-ENTROPY"
            
            # Log entropy for monitoring
            self.entropy_history.append(entropy)
            if training_mode and self.use_wandb:
                wandb.log({
                    "inference_entropy": entropy,
                    "inference_temperature": self.inference_temperature,
                    "step": self.step_count
                })
        elif training_mode:
            # Training mode: combine epsilon-greedy with temperature-based sampling
            if random.random() < self.epsilon:
                # Exploration: temperature-based sampling for diversity
                action, entropy = self._sample_action_with_temperature(state, self.temperature)
                exploration_mode = "EXPLORE-ENTROPY"
                self.entropy_history.append(entropy)
            else:
                # Exploitation: still use some temperature for diversity
                action, entropy = self._sample_action_with_temperature(state, self.temperature * 0.5)
                exploration_mode = "EXPLOIT-ENTROPY"
                self.entropy_history.append(entropy)
        else:
            # Standard inference mode: use softmax with low temperature
            action, entropy = self._sample_action_with_temperature(state, 0.5)
            exploration_mode = "INFERENCE"
            self.entropy_history.append(entropy)
        
        # Convert action to parameters
        total_tokens, depth, top_k = self._action_to_params(action)
        
        # Safety backup: clamp parameters if somehow invalid
        total_tokens, depth, top_k = self._safe_clamp_params(total_tokens, depth, top_k)
        
        # OPTIMIZATION 2: Update action cache
        if self.action_cache_enabled:
            self.cached_params = (total_tokens, depth, top_k)
            self.cached_action = action
            self.cache_step_counter = 1  # Reset counter
            self.cache_hidden_states = hidden_states
            
            # print(f"  New action predicted: {total_tokens}, {depth}, {top_k} - Cache reset to step 1")
            
            # Log cache update
            if training_mode and self.use_wandb:
                wandb.log({
                    "cache_update": 1,
                    "new_cached_params": {
                        "total_tokens": total_tokens,
                        "depth": depth,
                        "top_k": top_k
                    }
                })
        
        # Store state-action for learning
        if training_mode:
            self.last_state = state
            self.last_action = action
            self.last_params = (total_tokens, depth, top_k)
            self.last_exploration_mode = exploration_mode
            self.last_entropy = entropy
        
        self.step_count += 1
        
        return total_tokens, depth, top_k
    
    def _sample_action_with_temperature(self, state, temperature):
        """Sample action using temperature-based softmax for max-entropy exploration"""
        with torch.no_grad():
            # Ensure state tensor has correct dtype (float32) to match network weights
            state = state.float()
            q_values = self.q_network(state.unsqueeze(0))
            
            if temperature > 0:
                # Apply temperature scaling for max-entropy sampling
                scaled_q_values = q_values / temperature
                probs = torch.softmax(scaled_q_values, dim=-1)
                
                # Sample from the distribution
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()
                
                # Calculate entropy for monitoring
                entropy = dist.entropy().item()
                
                # Convert valid action index to actual action
                action = self.valid_actions[action_idx.item()]
                
                return action, entropy
            else:
                # Greedy action selection
                action_idx = torch.argmax(q_values, dim=-1)
                action = self.valid_actions[action_idx.item()]
                return action, 0.0
    
    def update_policy(self, reward, generation_time=None, new_tokens=None, training_mode=True):
        """Update policy with received reward using reward aggregation for cached actions"""
        # Convert reward to float to avoid tensor serialization issues
        reward_value = reward.item() if hasattr(reward, 'item') else float(reward)
        
        # Only do reward aggregation and policy updates in training mode
        if not training_mode:
            return
        
        # OPTIMIZATION: Reward aggregation for cached actions
        if self.action_cache_enabled:
            # Collect rewards during cache period
            self.cached_rewards.append(reward_value)
            if generation_time is not None:
                self.cached_generation_times.append(generation_time)
            if new_tokens is not None:
                self.cached_new_tokens.append(new_tokens)
            
            # Check if we should aggregate and update policy
            should_update = (
                self.last_state is not None and 
                self.last_action is not None and
                (self.cache_step_counter >= self.action_cache_steps or 
                 len(self.cached_rewards) >= self.action_cache_steps)
            )
            
            if should_update:
                # Aggregate rewards over the cache period
                if len(self.cached_rewards) > 0:
                    # Calculate aggregated metrics
                    avg_reward = sum(self.cached_rewards) / len(self.cached_rewards)
                    total_generation_time = sum(self.cached_generation_times) if self.cached_generation_times else 0
                    total_new_tokens = sum(self.cached_new_tokens) if self.cached_new_tokens else 0
                    
                    # Add experience to replay buffer with aggregated reward
                    valid_action_idx = self.valid_actions.index(self.last_action)
                    experience = (
                        self.last_state.clone().detach(),
                        valid_action_idx,
                        avg_reward,  # Use aggregated reward
                        None,  # next_state (episodic)
                        True   # done
                    )
                    self.memory.append(experience)
                    
                    # Add aggregated reward to history
                    self.reward_history.append(avg_reward)
                    
                    # Learn from batch if enough experiences
                    if len(self.memory) >= self.batch_size:
                        self._learn_from_batch()
                    
                    # Update target network periodically
                    if self.step_count % self.target_update_freq == 0:
                        self.target_network.load_state_dict(self.q_network.state_dict())
                    
                    # Decay epsilon
                    self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
                    
                    # Log aggregated metrics to wandb
                    if self.use_wandb:
                        log_data = {
                            "aggregated_reward": avg_reward,
                            "cache_period_steps": len(self.cached_rewards),
                            "step": self.step_count,
                            "avg_reward_100": sum(list(self.reward_history)[-100:]) / min(len(self.reward_history), 100),
                            "reward_std_in_cache": np.std(self.cached_rewards) if len(self.cached_rewards) > 1 else 0.0,
                            "epsilon": self.epsilon,
                            "exploration_mode": self.last_exploration_mode,
                            "entropy": self.last_entropy
                        }
                        
                        if total_generation_time > 0:
                            log_data["total_generation_time"] = total_generation_time
                            log_data["avg_generation_time"] = total_generation_time / len(self.cached_generation_times)
                        if total_new_tokens > 0:
                            log_data["total_new_tokens"] = total_new_tokens
                            log_data["avg_new_tokens"] = total_new_tokens / len(self.cached_new_tokens)
                            if total_generation_time > 0:
                                log_data["avg_tokens_per_second"] = total_new_tokens / total_generation_time
                        
                        # Add parameter info
                        if self.last_params:
                            log_data.update({
                                "total_tokens": self.last_params[0],
                                "depth": self.last_params[1],
                                "top_k": self.last_params[2]
                            })
                        
                        # Add loss if available
                        if len(self.loss_history) > 0:
                            log_data["loss"] = self.loss_history[-1]
                        
                        wandb.log(log_data)
                    
                    # Store cache period length before clearing
                    cache_period_length = len(self.cached_rewards)
                    
                    # Clear cache for next period
                    self.cached_rewards = []
                    self.cached_generation_times = []
                    self.cached_new_tokens = []
                    self.last_cache_update_step = self.step_count
                    
                    # print(f"🔄 Aggregated {cache_period_length} rewards: avg={avg_reward:.2f}, "
                        #   f"total_time={total_generation_time:.2f}s, total_tokens={total_new_tokens}")
        else:
            # Original behavior for non-cached mode
            if self.last_state is not None and self.last_action is not None:
                # Add experience to replay buffer
                valid_action_idx = self.valid_actions.index(self.last_action)
                experience = (
                    self.last_state.clone().detach(),
                    valid_action_idx,
                    reward_value,
                    None,  # next_state (episodic)
                    True   # done
                )
                self.memory.append(experience)
                
                # Add to reward history
                self.reward_history.append(reward_value)
                
                # Learn from batch if enough experiences
                if len(self.memory) >= self.batch_size:
                    self._learn_from_batch()
                
                # Update target network periodically
                if self.step_count % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                
                # Decay epsilon
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
                
                # Log to wandb
                if self.use_wandb:
                    log_data = {
                        "reward": reward_value,
                        "epsilon": self.epsilon,
                        "step": self.step_count,
                        "avg_reward_100": sum(list(self.reward_history)[-100:]) / min(len(self.reward_history), 100),
                        "exploration_mode": self.last_exploration_mode,
                        "entropy": self.last_entropy
                    }
                    
                    if generation_time is not None:
                        log_data["generation_time"] = generation_time
                    if new_tokens is not None:
                        log_data["new_tokens"] = new_tokens
                        if generation_time is not None and generation_time > 0:
                            log_data["tokens_per_second"] = new_tokens / generation_time
                    
                    # Add parameter info
                    if self.last_params:
                        log_data.update({
                            "total_tokens": self.last_params[0],
                            "depth": self.last_params[1],
                            "top_k": self.last_params[2]
                        })
                    
                    # Add loss if available
                    if len(self.loss_history) > 0:
                        log_data["loss"] = self.loss_history[-1]
                    
                    wandb.log(log_data)
        
        # Save checkpoint periodically
        if self.should_save_checkpoint():
            self.save_checkpoint()
            # Update last checkpoint step to prevent duplicate saves in the same step
            self.last_checkpoint_step = self.step_count
    
    def _clear_reward_cache(self, training_mode=True):
        """Clear reward cache with optional aggregation"""
        if not self.action_cache_enabled or len(self.cached_rewards) == 0:
            return None
        
        cache_period_length = len(self.cached_rewards)
        
        # Only aggregate if in training mode
        if training_mode:
            # Aggregate current cache
            avg_reward = sum(self.cached_rewards) / len(self.cached_rewards)
            total_generation_time = sum(self.cached_generation_times) if self.cached_generation_times else 0
            total_new_tokens = sum(self.cached_new_tokens) if self.cached_new_tokens else 0
            
            # Add experience to replay buffer with aggregated reward
            if self.last_state is not None and self.last_action is not None:
                valid_action_idx = self.valid_actions.index(self.last_action)
                experience = (
                    self.last_state.clone().detach(),
                    valid_action_idx,
                    avg_reward,  # Use aggregated reward
                    None,  # next_state (episodic)
                    True   # done
                )
                self.memory.append(experience)
                
                # Learn from batch if enough experiences
                if len(self.memory) >= self.batch_size:
                    self._learn_from_batch()
                
                # Update target network periodically
                if self.step_count % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                
                # Decay epsilon
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Add to reward history
            self.reward_history.append(avg_reward)
            
            # Log if wandb is enabled
            if self.use_wandb:
                log_data = {
                    "forced_aggregated_reward": avg_reward,
                    "cache_period_steps": cache_period_length,
                    "step": self.step_count,
                    "reward_std_in_cache": np.std(self.cached_rewards) if len(self.cached_rewards) > 1 else 0.0,
                    "epsilon": self.epsilon,
                    "exploration_mode": self.last_exploration_mode,
                    "entropy": self.last_entropy
                }
                
                if total_generation_time > 0:
                    log_data["total_generation_time"] = total_generation_time
                if total_new_tokens > 0:
                    log_data["total_new_tokens"] = total_new_tokens
                    if total_generation_time > 0:
                        log_data["avg_tokens_per_second"] = total_new_tokens / total_generation_time
                
                # Add loss if available
                if len(self.loss_history) > 0:
                    log_data["loss"] = self.loss_history[-1]
                
                wandb.log(log_data)
            
            # print(f"🔄 Aggregated {cache_period_length} rewards: avg={avg_reward:.2f}")
            
            result = {
                "avg_reward": avg_reward,
                "total_generation_time": total_generation_time,
                "total_new_tokens": total_new_tokens,
                "cache_period_length": cache_period_length
            }
        else:
            # print(f"🔄 Cleared {cache_period_length} cached rewards (inference mode)")
            result = None
        
        # Clear cache
        self._clear_cache_arrays()
        
        return result
    
    def _clear_cache_arrays(self):
        """Clear all cache arrays"""
        self.cached_rewards = []
        self.cached_generation_times = []
        self.cached_new_tokens = []
        self.last_cache_update_step = self.step_count
    
    def force_reward_aggregation(self, training_mode=True):
        """Force aggregation of currently cached rewards (useful for evaluation)"""
        return self._clear_reward_cache(training_mode)
    
    def reset(self, training_mode=True):
        """Reset policy state between different text generations"""
        # Clear any remaining cached rewards
        if self.action_cache_enabled and len(self.cached_rewards) > 0:
            self._clear_reward_cache(training_mode)
        else:
            # If no cached rewards, just clear cache arrays
            self._clear_cache_arrays()
        
        # Reset cache state
        self.cached_action = None
        self.cached_params = None
        self.cache_step_counter = 0
        self.cache_hidden_states = None
        
        # Reset training state
        self.last_state = None
        self.last_action = None
        self.last_params = None
        self.last_exploration_mode = None
        self.last_entropy = 0.0
        
        # print(f"🔄 Reset DQN policy state - cache cleared, step counter reset to 0")
    
    def _learn_from_batch(self):
        """Learn from a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (for episodic tasks, next_q_values = 0)
        target_q_values = rewards.unsqueeze(1)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Track loss
        self.loss_history.append(loss.item())
    
    def _precompute_valid_actions(self):
        """Precompute valid actions based on constraints"""
        valid_actions = []
        for action in range(self.total_actions):
            total_tokens, depth, top_k = self._action_to_params(action)
            if self._is_valid_combination(total_tokens, depth, top_k):
                valid_actions.append(action)
        return valid_actions
    
    def _is_valid_combination(self, total_tokens, depth, top_k):
        """Check if parameter combination satisfies constraints"""
        # Constraint: total_tokens <= top_k^(depth-1)
        max_tokens_constraint = top_k ** (depth - 1)
        return total_tokens <= max_tokens_constraint
    
    def _safe_clamp_params(self, total_tokens, depth, top_k):
        """Safely clamp parameters to valid ranges"""
        # Clamp to bin ranges
        total_tokens = max(min(total_tokens, max(self.total_tokens_bins)), min(self.total_tokens_bins))
        depth = max(min(depth, max(self.depth_bins)), min(self.depth_bins))
        top_k = max(min(top_k, max(self.top_k_bins)), min(self.top_k_bins))
        
        # Ensure constraint satisfaction
        max_allowed_tokens = top_k ** (depth - 1)
        if total_tokens > max_allowed_tokens:
            # Find closest valid total_tokens
            valid_tokens = [t for t in self.total_tokens_bins if t <= max_allowed_tokens]
            if valid_tokens:
                total_tokens = max(valid_tokens)
            else:
                total_tokens = min(self.total_tokens_bins)
        
        return total_tokens, depth, top_k
    
    def save_checkpoint(self, checkpoint_name=None):
        """Save model checkpoint"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{self.questions_processed}"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pth")
        
        # Update last checkpoint step to prevent duplicate saves
        self.last_checkpoint_step = self.step_count
        
        # Convert tensors to Python numbers for JSON/pickle serialization
        reward_history_serializable = []
        for reward in self.reward_history:
            if hasattr(reward, 'item'):  # PyTorch tensor
                reward_history_serializable.append(reward.item())
            else:
                reward_history_serializable.append(float(reward))
        
        loss_history_serializable = []
        for loss in self.loss_history:
            if hasattr(loss, 'item'):  # PyTorch tensor
                loss_history_serializable.append(loss.item())
            else:
                loss_history_serializable.append(float(loss))
        
        entropy_history_serializable = []
        for entropy in self.entropy_history:
            if hasattr(entropy, 'item'):  # PyTorch tensor
                entropy_history_serializable.append(entropy.item())
            else:
                entropy_history_serializable.append(float(entropy))
        
        # Save model state
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'reward_history': reward_history_serializable,
            'loss_history': loss_history_serializable,
            'entropy_history': entropy_history_serializable,
            'cache_step_counter': self.cache_step_counter,
            'cached_params': self.cached_params,
            'action_cache_enabled': self.action_cache_enabled,
            'action_cache_steps': self.action_cache_steps,
            'use_eagle3_features': self.use_eagle3_features,
            'hidden_size': self.hidden_size,
            # NEW: Reward aggregation state
            'cached_rewards': self.cached_rewards,
            'cached_generation_times': self.cached_generation_times,
            'cached_new_tokens': self.cached_new_tokens,
            'last_cache_update_step': self.last_cache_update_step,
            # NEW: Checkpoint tracking
            'last_checkpoint_step': self.last_checkpoint_step
        }, checkpoint_path)
        
        print(f"✅ Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load model checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model states
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            self.questions_processed = checkpoint.get('questions_processed', 0)
            self.training_seed = checkpoint.get('training_seed', None)
            self.step_count = checkpoint.get('step_count', 0)
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.reward_history = deque(checkpoint.get('reward_history', []), maxlen=1000)
            self.loss_history = deque(checkpoint.get('loss_history', []), maxlen=1000)
            self.entropy_history = deque(checkpoint.get('entropy_history', []), maxlen=1000)
            self.cache_step_counter = checkpoint.get('cache_step_counter', 0)
            self.cached_params = checkpoint.get('cached_params', None)
            # NEW: Restore reward aggregation state
            self.cached_rewards = checkpoint.get('cached_rewards', [])
            self.cached_generation_times = checkpoint.get('cached_generation_times', [])
            self.cached_new_tokens = checkpoint.get('cached_new_tokens', [])
            self.last_cache_update_step = checkpoint.get('last_cache_update_step', 0)
            # NEW: Restore checkpoint tracking
            self.last_checkpoint_step = checkpoint.get('last_checkpoint_step', -1)
            
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
            print(f"   Questions processed: {self.questions_processed}")
            print(f"   Step count: {self.step_count}")
            print(f"   Epsilon: {self.epsilon:.4f}")
            return True
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)), reverse=True)
        return os.path.join(self.checkpoint_dir, checkpoints[0])
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones"""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)), reverse=True)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[self.max_checkpoints:]:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
            try:
                os.remove(checkpoint_path)
                print(f"🗑️  Removed old checkpoint: {checkpoint}")
            except Exception as e:
                print(f"Warning: Could not remove {checkpoint}: {e}")
    
    def should_save_checkpoint(self):
        """Check if we should save a checkpoint"""
        # Only save if we haven't already saved for this step
        should_save = (self.step_count % self.checkpoint_freq == 0 and 
                      self.step_count != self.last_checkpoint_step)
        # if should_save:
        #     print(f"📄 Checkpoint save triggered: step={self.step_count}, last_saved={self.last_checkpoint_step}")
        return should_save
    
    def set_training_seed(self, seed):
        """Set training seed for reproducible shuffling"""
        self.training_seed = seed
    
    def get_resume_info(self):
        """Get information for resuming training"""
        return {
            "questions_processed": self.questions_processed,
            "training_seed": self.training_seed
        }
    
    def increment_questions_processed(self, count=1):
        """Increment the number of questions processed"""
        self.questions_processed += count
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if len(self.reward_history) == 0:
            return {
                "avg_reward": 0.0, 
                "reward_std": 0.0, 
                "total_steps": self.step_count,
                "cache_hit_rate": 0.0
            }
        
        rewards = list(self.reward_history)
        cache_hit_rate = (self.cache_step_counter / max(self.step_count, 1)) if self.action_cache_enabled else 0.0
        
        # Since rewards are now stored as floats, we can compute stats directly
        avg_reward = sum(rewards) / len(rewards)
        variance = sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)
        reward_std = variance ** 0.5
        
        # Loss and entropy histories should also contain floats now
        recent_losses = list(self.loss_history)[-100:]
        avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
        
        recent_entropies = list(self.entropy_history)[-100:]
        avg_entropy = sum(recent_entropies) / len(recent_entropies) if recent_entropies else 0.0
        
        stats = {
            "avg_reward": avg_reward,
            "reward_std": reward_std,
            "total_steps": self.step_count,
            "total_questions": self.questions_processed,
            "epsilon": self.epsilon,
            "avg_loss": avg_loss,
            "avg_entropy": avg_entropy,
            "cache_hit_rate": cache_hit_rate
        }
        
        # Add cache-specific statistics if action caching is enabled
        if self.action_cache_enabled:
            stats.update({
                "action_cache_enabled": True,
                "action_cache_steps": self.action_cache_steps,
                "current_cache_step": self.cache_step_counter,
                "pending_rewards": len(self.cached_rewards),
                "last_cache_update": self.last_cache_update_step
            })
            
            # Calculate cache efficiency metrics
            if self.step_count > 0:
                cache_efficiency = (self.step_count - len(self.reward_history)) / self.step_count
                stats["cache_efficiency"] = cache_efficiency
                
                # Estimate computational savings
                policy_calls_saved = self.step_count - len(self.reward_history)
                stats["policy_calls_saved"] = policy_calls_saved
                stats["computational_savings_pct"] = (policy_calls_saved / self.step_count) * 100
        
        return stats
    
    def save(self, path):
        """Save the entire policy"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'questions_processed': self.questions_processed,
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'action_cache_enabled': self.action_cache_enabled,
            'action_cache_steps': self.action_cache_steps,
            'use_eagle3_features': self.use_eagle3_features,
            'hidden_size': self.hidden_size,
            'valid_actions': self.valid_actions,
            'total_tokens_bins': self.total_tokens_bins,
            'depth_bins': self.depth_bins,
            'top_k_bins': self.top_k_bins
        }, path)
        
        print(f"💾 Saved optimized DQN policy to: {path}")
    
    def load(self, path):
        """Load the entire policy"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model states
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load metadata
            self.questions_processed = checkpoint.get('questions_processed', 0)
            self.step_count = checkpoint.get('step_count', 0)
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            
            print(f"📂 Loaded optimized DQN policy from: {path}")
            return True
        else:
            print(f"❌ Policy file not found: {path}")
            return False

def calculate_optimized_online_reward(generation_time, new_tokens, total_tokens, depth, top_k):
    """Calculate reward for optimized online learning"""
    if generation_time <= 0 or new_tokens <= 0:
        return 0.0
    
    # Reward is simply tokens per second
    tokens_per_second = new_tokens / generation_time
    return tokens_per_second
