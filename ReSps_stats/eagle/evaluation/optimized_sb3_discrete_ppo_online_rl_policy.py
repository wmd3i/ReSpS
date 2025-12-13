"""
Custom PPO Policy Implementation - No Stable Baselines 3 Dependency
1. Layer Feature Concatenation (EAGLE-3 features instead of SBERT)
2. Action Generation Frequency optimization (caching actions for N steps)
3. Custom PPO implementation following SB3 practices
"""

import numpy as np
import torch
import torch.nn as nn
import os
import json
import contextlib
from collections import deque
import random
import gym
from gym import spaces
from typing import Tuple

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Import our custom PPO components
from eagle.evaluation.custom_ppo_components import (
    ActorCriticPolicy, 
    CustomPPO, 
    BaseCallback, 
    CheckpointCallback,
    RolloutBuffer
)

class CustomPolicy(ActorCriticPolicy):
    """Custom Policy Network with enhanced architecture for EAGLE parameter optimization"""
    
    def __init__(self, observation_space, action_space, net_arch=None, **kwargs):
        if net_arch is None:
            net_arch = [64, 64]  # Enhanced architecture for better performance
        
        super().__init__(observation_space, action_space, net_arch=net_arch, **kwargs)
        
        # Add additional layers for better feature extraction
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.features_dim, self.features_dim),
            nn.LayerNorm(self.features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Override the shared network with enhanced version
        enhanced_layers = []
        prev_dim = self.features_dim
        for layer_size in net_arch:
            enhanced_layers.extend([
                nn.Linear(prev_dim, layer_size),
                nn.LayerNorm(layer_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = layer_size
        
        self.shared_net = nn.Sequential(*enhanced_layers)
        
        # Enhanced policy and value heads
        self.policy_net = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.action_dim)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced forward pass with feature enhancement"""
        enhanced_features = self.feature_enhancer(obs)
        features = self.shared_net(enhanced_features)
        policy_logits = self.policy_net(features)
        value = self.value_net(features)
        return policy_logits, value
    
class OptimizedEagleParameterEnv(gym.Env):
    """Optimized Custom Gym environment for EAGLE parameter optimization
    
    Optimizations:
    1. Uses EAGLE-3 layer features instead of SBERT text embeddings
    2. Supports action caching to reduce computation frequency
    3. NEW: Context-only state representation (SBERT embeddings without projection)
    """
    
    def __init__(self, hidden_size=4096, use_context_only_state=False):
        super(OptimizedEagleParameterEnv, self).__init__()
        
        # Parameter bins (6×6×5 = 180 total combinations)
        self.total_tokens_bins = [32, 48, 64, 80, 96, 128]  # 6 options
        self.depth_bins = [3, 4, 5, 6, 7, 8]  # 6 options  
        self.top_k_bins = [8, 12, 16, 20, 32]  # 5 options
        
        # Action space dimensions
        self.n_total_tokens = len(self.total_tokens_bins)
        self.n_depth = len(self.depth_bins)
        self.n_top_k = len(self.top_k_bins)
        self.total_actions = self.n_total_tokens * self.n_depth * self.n_top_k
        
        # Precompute valid actions to avoid constraint violations
        self.valid_actions = self._precompute_valid_actions()
        
        # NEW: Context-only state representation option
        self.use_context_only_state = use_context_only_state
        
        if use_context_only_state:
            # Use SBERT embeddings directly (384 dimensions)
            self.feature_dim = 384
            self.hidden_size = 384  # Match SBERT output dimension
            state_description = "SBERT context embeddings (384D)"
        else:
            # OPTIMIZATION 1: Use EAGLE-3 layer features instead of SBERT
            # The hidden_size is k (model hidden size), and we expect 3k features from EAGLE-3
            self.hidden_size = hidden_size
            self.feature_dim = hidden_size  # After FC layer reduction: 3k -> k
            state_description = "EAGLE-3 layer features"
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.valid_actions))  # Only valid actions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.feature_dim,), dtype=np.float32
        )
        
        # Environment state
        self.current_hidden_states = None
        self.current_context = None
        self.step_count = 0
        
        print(f"OptimizedEagleParameterEnv initialized:")
        print(f"  - Total parameter combinations: {self.total_actions}")
        print(f"  - Valid parameter combinations: {len(self.valid_actions)}")
        print(f"  - Valid coverage: {len(self.valid_actions)/self.total_actions*100:.1f}%")
        print(f"  - State representation: {state_description}")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Use context-only state: {use_context_only_state}")
        print(f"  - Constraint: total_tokens ≤ top_k^(depth-1)")
    
    def _precompute_valid_actions(self):
        """Precompute valid action indices based on constraint: total_tokens <= top_k^(depth-1)"""
        valid_actions = []
        for action in range(self.total_actions):
            total_tokens, depth, top_k = self._action_to_params(action)
            if self._is_valid_combination(total_tokens, depth, top_k):
                valid_actions.append(action)
        return valid_actions
    
    def _is_valid_combination(self, total_tokens, depth, top_k):
        """Check if parameter combination satisfies constraints"""
        max_tokens_constraint = top_k ** (depth - 1)
        basic_constraint = total_tokens <= max_tokens_constraint
        return basic_constraint
    
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
    
    def _encode_state_from_hidden_states(self, hidden_states):
        """OPTIMIZATION 1: Encode state from EAGLE-3 layer features instead of SBERT
        
        Args:
            hidden_states: Concatenated hidden states from EAGLE-3 (3k dimensions)
                          or already reduced features (k dimensions)
        
        Returns:
            numpy array of shape (feature_dim,) ready for RL policy
        """
        if hidden_states is None:
            # Return zero state if no hidden states available
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Convert to numpy and ensure correct shape
        if isinstance(hidden_states, torch.Tensor):
            features = hidden_states.detach().cpu().numpy()
        else:
            features = np.array(hidden_states)
        
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
            features = features.reshape(3, self.hidden_size).mean(axis=0)
        elif features.shape[-1] == self.hidden_size:
            # This is already the k-dimensional reduced features
            pass
        else:
            # Unexpected dimension, pad or truncate to match expected size
            if features.shape[-1] > self.hidden_size:
                features = features[:self.hidden_size]
            else:
                padded = np.zeros(self.hidden_size)
                padded[:features.shape[-1]] = features
                features = padded
        
        return features.astype(np.float32)
    
    def _encode_state_from_context(self, context):
        """NEW: Encode state directly from context using SBERT embeddings (384D)
        
        Args:
            context: Text context string
        
        Returns:
            numpy array of shape (384,) - SBERT embeddings without projection
        """
        if not self.use_context_only_state:
            raise ValueError("Context-only state encoding is not enabled. Set use_context_only_state=True")
        
        if context is None or context == "":
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Initialize SBERT model if needed
        if not hasattr(self, '_sbert_model'):
            try:
                from sentence_transformers import SentenceTransformer
                self._sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("🔧 Initialized SBERT model for context-only state representation")
            except ImportError:
                raise ImportError("sentence-transformers is required for context-only state. Install with: pip install sentence-transformers")
        
        # Encode context directly to 384D - WITH OVERHEAD MEASUREMENT
        try:
            from eagle.evaluation.gen_ea_answer_llama3chat_rl_measure import overhead_profiler
        except ImportError:
            # Fallback: create a dummy profiler if import fails
            class DummyProfiler:
                def time_component(self, name):
                    return DummyContext()
            
            class DummyContext:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            
            overhead_profiler = DummyProfiler()
        
        with overhead_profiler.time_component("SentenceBERT_Text_to_Embedding"):
            with torch.no_grad() if hasattr(self, '_sbert_model') else contextlib.nullcontext():
                sbert_embedding = self._sbert_model.encode(context)  # Shape: (384,)
        
        return sbert_embedding.astype(np.float32)
    
    def reset(self):
        """Reset environment state"""
        self.current_hidden_states = None
        self.current_context = None
        self.step_count = 0
        # Return zero state - will be set properly when predict_parameters is called
        return np.zeros(self.feature_dim, dtype=np.float32)
    
    def step(self, action):
        """Execute one step with the given action"""
        # Convert from valid action index to actual action
        actual_action = self.valid_actions[action]
        
        # Convert action to parameters
        total_tokens, depth, top_k = self._action_to_params(actual_action)
        
        # Store last parameters for external reward calculation
        self.last_params = (total_tokens, depth, top_k)
        
        # The reward will be set externally by the policy
        reward = 0.0  # Placeholder
        done = True   # Each inference is episodic
        info = {
            'total_tokens': total_tokens,
            'depth': depth,
            'top_k': top_k,
            'valid_action': actual_action
        }
        
        # Don't increment step_count here - it's handled by the policy
        
        # Return next observation (will be updated externally)
        next_obs = np.zeros(self.feature_dim, dtype=np.float32)
        
        return next_obs, reward, done, info

class CustomPPOOnlineTreePolicy:
    """Custom PPO-based Online RL Policy - No Stable Baselines 3 Dependency
    
    Optimizations:
    1. Uses EAGLE-3 layer features instead of SBERT text embeddings
    2. Action caching to reduce computation frequency (generate action every N steps)
    3. Custom PPO implementation following SB3 practices
    
    Supports both standard PPO and max-entropy PPO modes:
    - Standard PPO: Low entropy coefficient, deterministic inference
    - Max-Entropy PPO: High entropy coefficient, temperature-based inference
    """
    
    def __init__(self, 
                 learning_rate=3e-4,
                 n_steps=64,
                 batch_size=32,
                 n_epochs=4,
                 gamma=0.95,
                 gae_lambda=0.9,
                 clip_range=0.2,
                 ent_coef=0.1,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 # Max-entropy specific parameters
                 enable_max_entropy=True,
                 max_entropy_ent_coef=0.1,
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
                 # NEW: Network architecture parameter
                 net_arch=None,                # Network architecture for policy network
                 use_wandb=True,
                 wandb_project="eagle-custom-ppo",
                 wandb_run_name=None,
                 checkpoint_dir="custom_ppo_checkpoints",
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
        self.enable_max_entropy = enable_max_entropy
        self.max_entropy_ent_coef = max_entropy_ent_coef
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
            wandb_run_name = wandb_run_name or f"optimized-eagle-sb3-ppo-{int(torch.randint(0, 10000, (1,)).item())}"
            wandb.init(project=wandb_project, name=wandb_run_name, reinit=True)
            wandb.config.update({
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "enable_max_entropy": enable_max_entropy,
                "action_cache_steps": action_cache_steps,
                "action_cache_enabled": action_cache_enabled,
                "use_eagle3_features": use_eagle3_features,
                "use_context_only_state": use_context_only_state,
                "hidden_size": self.hidden_size
            })
        else:
            print("Warning: Wandb logging disabled")
        
        # Initialize environment with optimizations
        self.env = OptimizedEagleParameterEnv(
            hidden_size=self.hidden_size, 
            use_context_only_state=use_context_only_state
        )
        
        # Determine entropy coefficient based on mode
        actual_ent_coef = max_entropy_ent_coef if enable_max_entropy else ent_coef
        
        # Store network architecture for policy initialization
        self.net_arch = net_arch if net_arch is not None else [64, 64]
        
        # Initialize custom policy with network architecture
        self.policy = CustomPolicy(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            net_arch=self.net_arch
        )
        
        # Initialize custom PPO model
        self.model = CustomPPO(
            policy=self.policy,
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=actual_ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=None,
            device=self.device
        )
        
        # Initialize training tracking
        self.reward_history = deque(maxlen=1000)
        self.last_state = None
        self.last_action = None
        self.last_params = None
        self.step_count = 0
        
        # PPO learning state
        self.current_obs = self.env.reset()
        self.episode_rewards = []
        self.episode_length = 0
        self.rollout_buffer = []
        
        print(f"CustomPPOOnlineTreePolicy initialized:")
        print(f"  - Action space: {self.env.action_space.n} discrete actions")
        print(f"  - Observation space: {self.env.observation_space.shape}")
        print(f"  - Network architecture: {self.net_arch}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Custom PPO n_steps: {n_steps}")
        print(f"  - Custom PPO batch_size: {batch_size}")
        print(f"  - Custom PPO n_epochs: {n_epochs}")
        print(f"  - Entropy coefficient: {actual_ent_coef} ({'high for diversity' if enable_max_entropy else 'standard'})")
        print(f"  - OPTIMIZATION 1 - EAGLE-3 features: {'enabled' if use_eagle3_features else 'disabled'}")
        print(f"  - OPTIMIZATION 2 - Action caching: {'enabled' if action_cache_enabled else 'disabled'}")
        if action_cache_enabled:
            print(f"    - Cache steps: {action_cache_steps} (generate action every {action_cache_steps} steps)")
        if enable_max_entropy:
            print(f"  - Inference temperature: {self.inference_temperature}")
            print(f"  - Max-entropy inference: {self.max_entropy_inference}")
        else:
            print(f"  - Inference mode: Deterministic/Standard Stochastic")
        print(f"  - Device: {self.device}")
        print(f"  - Wandb logging: {'enabled' if self.use_wandb else 'disabled'}")
        
        if enable_max_entropy:
            print(f"🌟 CUSTOM MAX-ENTROPY MODE: EAGLE-3 features + action caching + higher exploration")
        else:
            print(f"⚙️  CUSTOM STANDARD PPO MODE: EAGLE-3 features + action caching + standard exploration")
    
    def predict_parameters(self, context=None, hidden_states=None, training_mode=True):
        """OPTIMIZED: Predict parameters using EAGLE-3 features with action caching
        
        Args:
            context: Text context (used for context-only state or fallback)
            hidden_states: EAGLE-3 layer features (used when use_context_only_state=False)
            training_mode: Whether in training mode
        
        Returns:
            tuple: (total_tokens, depth, top_k)
        """
        # NEW: Handle context-only state mode
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
            state = self.env._encode_state_from_context(context)
            self.env.current_context = context
            feature_source = "CONTEXT_SBERT"
            
            # Track usage statistics
            if training_mode and self.use_wandb:
                wandb.log({
                    "feature_source": "context_sbert",
                    "context_length": len(context) if context else 0,
                    "state_dimension": 384
                })
        
        # Original EAGLE-3 features mode 
        elif hidden_states is None:
            return 96, 8, 20  # Default parameters
        else:
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
            
            # OPTIMIZATION 1: Use EAGLE-3 features if available
            if self.use_eagle3_features and hidden_states is not None:
                # PRIMARY MODE: Use EAGLE-3 layer features (dense 4096-dim representation)
                state = self.env._encode_state_from_hidden_states(hidden_states)
                # Store hidden states for future reference
                self.env.current_hidden_states = hidden_states
                feature_source = "EAGLE3"
                
                # Track usage statistics
                if training_mode and self.use_wandb:
                    wandb.log({
                        "feature_source": "eagle3",
                        "sbert_usage": 0,
                        "eagle3_usage": 1
                    })
            else:
                # FALLBACK MODE: Map SBERT features to same 4096-dim space with proper scaling
                if context is not None:
                    from sentence_transformers import SentenceTransformer
                    if not hasattr(self, '_sbert_model'):
                        self._sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    with overhead_profiler.time_component("SentenceBERT_Fallback_Encoding"):
                        sbert_embedding = self._sbert_model.encode(context)  # Shape: (384,)
                    
                    # SOLUTION: Intelligent feature mapping instead of zero-padding
                    # Use learned projection to maintain feature density
                    if not hasattr(self, '_feature_mapper'):
                        # Create a learnable mapping from SBERT (384) to EAGLE-3 space (4096)
                        self._feature_mapper = torch.nn.Sequential(
                            torch.nn.Linear(384, self.env.feature_dim),
                            torch.nn.LayerNorm(self.env.feature_dim),  # Normalize to match EAGLE-3 scale
                            torch.nn.Tanh()  # Keep values in reasonable range
                        ).to(self.device)
                        
                        # Initialize with small weights to start conservative
                        with torch.no_grad():
                            for module in self._feature_mapper:
                                if isinstance(module, torch.nn.Linear):
                                    torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                                    torch.nn.init.zeros_(module.bias)

                        print(f"🔧 Created SBERT→EAGLE-3 feature mapper (384→{self.env.feature_dim})")
                        print("   This avoids sparse padding and maintains feature density")
                    
                    # Map SBERT features to EAGLE-3 compatible space
                    with torch.no_grad():
                        sbert_tensor = torch.FloatTensor(sbert_embedding).unsqueeze(0).to(self.device)
                        mapped_features = self._feature_mapper(sbert_tensor).squeeze(0)
                        state = mapped_features.cpu().numpy()
                    
                    feature_source = "SBERT_MAPPED"
                    
                    # Log the mapping for monitoring
                    if not hasattr(self, '_mapping_warned'):
                        print(f"⚠️  Using SBERT fallback with learned mapping:")
                        print(f"   Input: SBERT {sbert_embedding.shape} → Output: {state.shape}")
                        print(f"   Feature density: 100% (no zero padding)")
                        print(f"   Recommendation: Use EAGLE-3 features for best performance")
                        self._mapping_warned = True
                    
                    # Track usage statistics
                    if training_mode and self.use_wandb:
                        wandb.log({
                            "feature_source": "sbert_mapped",
                            "sbert_usage": 1,
                            "eagle3_usage": 0
                        })
                        
                else:
                    # No input available, use zero state (matching EAGLE-3 dimension)
                    state = np.zeros(self.env.feature_dim, dtype=np.float32)
                    feature_source = "ZERO"
                    return 96, 8, 20  # Default parameters if no context or hidden states
        self.env.current_context = context if context else ""
        
        # Import overhead profiler for timing measurements
        try:
            from eagle.evaluation.gen_ea_answer_llama3chat_rl_measure import overhead_profiler
        except ImportError:
            class DummyProfiler:
                def time_component(self, name):
                    return self
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            overhead_profiler = DummyProfiler()
        
        # Choose prediction strategy based on mode and phase
        if self.enable_max_entropy and self.max_entropy_inference and not training_mode:
            # MAX-ENTROPY INFERENCE: Use temperature-based sampling for diversity
            if self.inference_temperature != 1.0:
                try:
                    with overhead_profiler.time_component("RL_Policy_Prediction_Custom_Max_Entropy_Temperature"):
                        action = self._sample_with_temperature(state, self.inference_temperature)
                    exploration_mode = "MAX-ENTROPY"
                except Exception as e:
                    print(f"⚠️  Temperature sampling failed: {e}")
                    print("   Using enhanced stochastic sampling instead")
                    with overhead_profiler.time_component("RL_Policy_Prediction_Custom_Enhanced_Stochastic"):
                        action = self._enhanced_stochastic_sampling(state)
                    exploration_mode = "ENHANCED-STOCHASTIC"
            else:
                with overhead_profiler.time_component("RL_Policy_Prediction_Custom_Max_Entropy_Stochastic"):
                    action, _ = self.model.predict(state, deterministic=False) 
                exploration_mode = "STOCHASTIC"
        elif training_mode:
            # TRAINING MODE: Always use stochastic for both modes (exploration)
            with overhead_profiler.time_component("RL_Policy_Prediction_Custom_Training"):
                action, _ = self.model.predict(state, deterministic=False)
            exploration_mode = "EXPLORE"
        else:
            # STANDARD INFERENCE: Use deterministic prediction
            with overhead_profiler.time_component("RL_Policy_Prediction_Custom_Deterministic"):
                action, _ = self.model.predict(state, deterministic=True)
            exploration_mode = "DETERMINISTIC"
        
        # Convert to actual parameters
        actual_action = self.env.valid_actions[action]
        total_tokens, depth, top_k = self.env._action_to_params(actual_action)
        
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
        
        # Store for training - UPDATE THE ENVIRONMENT STATE
        if training_mode:
            # Update environment observation for PPO learning
            self.current_obs = state.copy()
            self.env.current_hidden_states = hidden_states
            self.env.current_context = context if context else ""
            
            self.last_state = state
            self.last_action = action
            self.last_params = (total_tokens, depth, top_k)
        
        self.step_count += 1
        
        return total_tokens, depth, top_k
    
    def _sample_with_temperature(self, state, temperature):
        """Sample action using temperature-based softmax for max-entropy exploration"""
        obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            try:
                # Get logits from our custom policy network
                policy_logits, _ = self.model.policy(obs_tensor)
                
                # Apply temperature scaling
                if temperature > 0:
                    scaled_logits = policy_logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Sample from the distribution
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    
                    return action.item()
                else:
                    return torch.argmax(policy_logits, dim=-1).item()
                    
            except Exception as e:
                print(f"Temperature sampling error: {e}")
                # Fallback to standard prediction
                action, _ = self.model.predict(state, deterministic=False)
                return action
    
    def _enhanced_stochastic_sampling(self, state):
        """Enhanced stochastic sampling as fallback"""
        action, _ = self.model.predict(state, deterministic=False)
        return action
    
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
                    
                    # ADD PROPER PPO LEARNING: Step the environment with the reward
                    if self.last_state is not None and self.last_action is not None:
                        # Step the environment to provide the reward and get done signal
                        next_obs, _, done, info = self.env.step(self.last_action)
                        
                        # Store the experience in rollout buffer for PPO learning
                        experience = {
                            'obs': self.current_obs,
                            'action': self.last_action,
                            'reward': avg_reward,
                            'next_obs': next_obs,
                            'done': done
                        }
                        self.rollout_buffer.append(experience)
                        self.episode_rewards.append(avg_reward)
                        self.episode_length += 1
                        
                        # Update current observation for next prediction
                        self.current_obs = self.env.reset()  # Reset for next episode (episodic task)
                        
                        # Trigger PPO learning when we have enough data
                        if len(self.rollout_buffer) >= self.model.n_steps:
                            # Learn from collected experiences
                            self._trigger_ppo_learning()
                            # Clear the rollout buffer
                            self.rollout_buffer = []
                    
                    # Add aggregated reward to history
                    self.reward_history.append(avg_reward)
                    
                    # Log aggregated metrics to wandb
                    if self.use_wandb:
                        log_data = {
                            "aggregated_reward": avg_reward,
                            "cache_period_steps": len(self.cached_rewards),
                            "step": self.step_count,
                            "avg_reward_100": sum(list(self.reward_history)[-100:]) / min(len(self.reward_history), 100),
                            "reward_std_in_cache": np.std(self.cached_rewards) if len(self.cached_rewards) > 1 else 0.0,
                            "episode_length": self.episode_length,
                            "rollout_buffer_size": len(self.rollout_buffer)
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
                        
                        wandb.log(log_data)
                    
                    # Store cache period length before clearing
                    cache_period_length = len(self.cached_rewards)
                    
                    # Clear cache for next period
                    self.cached_rewards = []
                    self.cached_generation_times = []
                    self.cached_new_tokens = []
                    self.last_cache_update_step = self.step_count
                    
                    # Reset episode tracking
                    self.episode_rewards = []
                    self.episode_length = 0
                    
                    # print(f"🔄 Aggregated {cache_period_length} rewards: avg={avg_reward:.2f}, "
                        #   f"total_time={total_generation_time:.2f}s, total_tokens={total_new_tokens}")
        else:
            # Original behavior for non-cached mode - ALSO ADD PPO LEARNING
            if self.last_state is not None and self.last_action is not None:
                # Step the environment to provide the reward
                next_obs, _, done, info = self.env.step(self.last_action)
                
                # Store the experience in rollout buffer for PPO learning
                experience = {
                    'obs': self.current_obs,
                    'action': self.last_action,
                    'reward': reward_value,
                    'next_obs': next_obs,
                    'done': done
                }
                self.rollout_buffer.append(experience)
                self.episode_rewards.append(reward_value)
                self.episode_length += 1
                
                # Update current observation for next prediction
                self.current_obs = self.env.reset()  # Reset for next episode (episodic task)
                
                # Trigger PPO learning when we have enough data
                if len(self.rollout_buffer) >= self.model.n_steps:
                    # Learn from collected experiences
                    self._trigger_ppo_learning()
                    # Clear the rollout buffer
                    self.rollout_buffer = []
                
                # Add to reward history
                self.reward_history.append(reward_value)
                
                # Log to wandb
                if self.use_wandb:
                    log_data = {
                        "reward": reward_value,
                        "step": self.step_count,
                        "avg_reward_100": sum(list(self.reward_history)[-100:]) / min(len(self.reward_history), 100),
                        "episode_length": self.episode_length,
                        "rollout_buffer_size": len(self.rollout_buffer)
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
                    
                    wandb.log(log_data)
                
                # Reset episode tracking
                self.episode_rewards = []
                self.episode_length = 0
        
        # Save checkpoint periodically
        if self.should_save_checkpoint():
            self.save_checkpoint()
            # Update last checkpoint step to prevent duplicate saves in the same step
            self.last_checkpoint_step = self.step_count
    
    def _trigger_ppo_learning(self):
        """Trigger PPO learning from collected rollout experiences"""
        if len(self.rollout_buffer) == 0:
            return
        
        try:
            print(f"🎯 Triggering PPO learning with {len(self.rollout_buffer)} experiences")
            
            # Extract observations, actions, and rewards from rollout buffer
            observations = np.array([exp['obs'] for exp in self.rollout_buffer])
            actions = np.array([exp['action'] for exp in self.rollout_buffer])
            rewards = np.array([exp['reward'] for exp in self.rollout_buffer])
            
            # Use our custom PPO's manual update method
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            
            # Perform one gradient step
            self._manual_ppo_update(obs_tensor, actions_tensor, rewards_tensor)
            
            if self.use_wandb:
                wandb.log({
                    "ppo_learning_triggered": 1,
                    "rollout_size": len(self.rollout_buffer),
                    "avg_rollout_reward": np.mean(rewards),
                    "learning_step": self.step_count
                })
                
        except Exception as e:
            print(f"Warning: PPO learning failed: {e}")
            # Continue without learning rather than crashing
            if self.use_wandb:
                wandb.log({"ppo_learning_error": 1})
    
    def _manual_ppo_update(self, observations, actions, rewards):
        """Manually perform a PPO update step using our custom implementation"""
        try:
            # Get policy and value function outputs
            with torch.no_grad():
                values, log_probs, entropy = self.model.policy.evaluate_actions(observations, actions)
                old_log_probs = log_probs.detach()
            
            # Calculate advantages (simplified - normally would use GAE)
            advantages = rewards - values
            returns = rewards
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update for several epochs
            for epoch in range(self.model.n_epochs):
                # Get current policy outputs
                current_values, new_log_probs, current_entropy = self.model.policy.evaluate_actions(observations, actions)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.model.clip_range, 1 + self.model.clip_range)
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(current_values, returns)
                
                # Total loss
                loss = policy_loss + self.model.vf_coef * value_loss - self.model.ent_coef * current_entropy.mean()
                
                # Gradient step
                self.model.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.model.max_grad_norm)
                self.model.optimizer.step()
            
            # Log learning metrics
            if self.use_wandb:
                wandb.log({
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": current_entropy.mean().item(),
                    "total_loss": loss.item(),
                    "mean_advantage": advantages.mean().item(),
                    "mean_return": returns.mean().item()
                })
                
            print(f"   PPO update completed at step {self.step_count} - Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
            
        except Exception as e:
            print(f"   PPO manual update failed: {e}")
            raise
    
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
            
            # ADD PROPER PPO LEARNING: Step the environment with the reward
            if self.last_state is not None and self.last_action is not None:
                # Step the environment to provide the reward and get done signal
                next_obs, _, done, info = self.env.step(self.last_action)
                
                # Store the experience in rollout buffer for PPO learning
                experience = {
                    'obs': self.current_obs,
                    'action': self.last_action,
                    'reward': avg_reward,
                    'next_obs': next_obs,
                    'done': done
                }
                self.rollout_buffer.append(experience)
                
                # Update current observation for next prediction
                self.current_obs = self.env.reset()  # Reset for next episode (episodic task)
                
                # Trigger PPO learning when we have enough data
                if len(self.rollout_buffer) >= self.model.n_steps:
                    # Learn from collected experiences
                    self._trigger_ppo_learning()
                    # Clear the rollout buffer
                    self.rollout_buffer = []
            
            # Add to reward history
            self.reward_history.append(avg_reward)
            
            # Log if wandb is enabled
            if self.use_wandb:
                log_data = {
                    "forced_aggregated_reward": avg_reward,
                    "cache_period_steps": cache_period_length,
                    "step": self.step_count,
                    "reward_std_in_cache": np.std(self.cached_rewards) if len(self.cached_rewards) > 1 else 0.0,
                }
                
                if total_generation_time > 0:
                    log_data["total_generation_time"] = total_generation_time
                if total_new_tokens > 0:
                    log_data["total_new_tokens"] = total_new_tokens
                    if total_generation_time > 0:
                        log_data["avg_tokens_per_second"] = total_new_tokens / total_generation_time
                
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
        
        # Reset PPO learning state
        self.current_obs = self.env.reset() if hasattr(self, 'env') else None
        self.episode_rewards = []
        self.episode_length = 0
        # Don't clear rollout_buffer here - let it accumulate across episodes
        
        # Reset environment state
        if hasattr(self, 'env'):
            self.env.reset()
        
        # print(f"🔄 Reset Custom PPO policy state - cache cleared, step counter reset to 0")
    
    def save_checkpoint(self, checkpoint_name=None):
        """Save model checkpoint"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{self.questions_processed}"
        
        # Use .pt extension for custom PPO (non-OFL version)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}.pt")
        
        # Save custom PPO model
        self.model.save(checkpoint_path)
        
        # Update last checkpoint step to prevent duplicate saves
        self.last_checkpoint_step = self.step_count
        
        # Save additional state
        state_path = os.path.join(self.checkpoint_dir, f"{checkpoint_name}_state.json")
        
        # Convert tensors to Python numbers for JSON serialization
        reward_history_serializable = []
        for reward in self.reward_history:
            if hasattr(reward, 'item'):  # PyTorch tensor
                reward_history_serializable.append(reward.item())
            else:
                reward_history_serializable.append(float(reward))
        
        # Convert cached_new_tokens tensors to Python numbers
        cached_new_tokens_serializable = []
        for tokens in self.cached_new_tokens:
            if hasattr(tokens, 'item'):  # PyTorch tensor
                cached_new_tokens_serializable.append(tokens.item())
            else:
                cached_new_tokens_serializable.append(int(tokens))
        
        state = {
            "questions_processed": self.questions_processed,
            "training_seed": self.training_seed,
            "step_count": self.step_count,
            "reward_history": reward_history_serializable,
            "cache_step_counter": self.cache_step_counter,
            "cached_params": self.cached_params,
            "enable_max_entropy": self.enable_max_entropy,
            "action_cache_enabled": self.action_cache_enabled,
            "action_cache_steps": self.action_cache_steps,
            "use_eagle3_features": self.use_eagle3_features,
            # NEW: Reward aggregation state
            "cached_rewards": self.cached_rewards,
            "cached_generation_times": self.cached_generation_times,
            "cached_new_tokens": cached_new_tokens_serializable,
            "last_cache_update_step": self.last_cache_update_step,
            # PPO learning state
            "episode_length": self.episode_length,
            "rollout_buffer_size": len(self.rollout_buffer),
            # NEW: Checkpoint tracking
            "last_checkpoint_step": self.last_checkpoint_step
        }
        
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"✅ Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load model checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Load custom PPO model
            self.model.load(checkpoint_path)
            
            # Load additional state
            state_path = checkpoint_path.replace('.pt', '_state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.questions_processed = state.get("questions_processed", 0)
                self.training_seed = state.get("training_seed", None)
                self.step_count = state.get("step_count", 0)
                self.reward_history = deque(state.get("reward_history", []), maxlen=1000)
                self.cache_step_counter = state.get("cache_step_counter", 0)
                self.cached_params = tuple(state["cached_params"]) if state.get("cached_params") else None
                # NEW: Restore reward aggregation state
                self.cached_rewards = state.get("cached_rewards", [])
                self.cached_generation_times = state.get("cached_generation_times", [])
                # Restore cached_new_tokens (already deserialized as Python numbers)
                self.cached_new_tokens = state.get("cached_new_tokens", [])
                self.last_cache_update_step = state.get("last_cache_update_step", 0)
                
                # Restore PPO learning state
                self.episode_length = state.get("episode_length", 0)
                # Don't restore rollout buffer - start fresh
                self.rollout_buffer = []
                self.episode_rewards = []
                self.current_obs = self.env.reset()
                # NEW: Restore checkpoint tracking
                self.last_checkpoint_step = state.get("last_checkpoint_step", -1)
            
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
            print(f"   Questions processed: {self.questions_processed}")
            print(f"   Step count: {self.step_count}")
            return True
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)), reverse=True)
        return os.path.join(self.checkpoint_dir, checkpoints[0])
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones"""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)), reverse=True)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[self.max_checkpoints:]:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
            state_path = checkpoint_path.replace('.pt', '_state.json')
            
            try:
                os.remove(checkpoint_path)
                if os.path.exists(state_path):
                    os.remove(state_path)
                print(f"🗑️  Removed old checkpoint: {checkpoint}")
            except Exception as e:
                print(f"Warning: Could not remove {checkpoint}: {e}")
    
    def should_save_checkpoint(self):
        """Check if we should save a checkpoint"""
        # Only save if we haven't already saved for this step
        should_save = (self.step_count % self.checkpoint_freq == 0 and 
                      self.step_count != self.last_checkpoint_step)
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
            return {"avg_reward": 0.0, "reward_std": 0.0, "total_steps": self.step_count}
        
        rewards = list(self.reward_history)
        # Since rewards are now stored as floats, we can compute stats directly
        avg_reward = sum(rewards) / len(rewards)
        variance = sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)
        reward_std = variance ** 0.5
        
        stats = {
            "avg_reward": avg_reward,
            "reward_std": reward_std,
            "total_steps": self.step_count,
            "total_questions": self.questions_processed,
            "cache_hit_rate": (self.cache_step_counter / max(self.step_count, 1)) if self.action_cache_enabled else 0.0
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
        self.model.save(path)
        
        # Save additional metadata (use .pt extension for custom PPO)
        metadata_path = path.replace('.pt', '_metadata.json')
        metadata = {
            "questions_processed": self.questions_processed,
            "step_count": self.step_count,
            "enable_max_entropy": self.enable_max_entropy,
            "action_cache_enabled": self.action_cache_enabled,
            "action_cache_steps": self.action_cache_steps,
            "use_eagle3_features": self.use_eagle3_features,
            "hidden_size": self.hidden_size,
            "last_checkpoint_step": self.last_checkpoint_step
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved custom PPO policy to: {path}")
    
    def load(self, path):
        """Load the entire policy"""
        if os.path.exists(path):
            self.model.load(path)
            
            # Load metadata if available
            metadata_path = path.replace('.pt', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.questions_processed = metadata.get("questions_processed", 0)
                self.step_count = metadata.get("step_count", 0)
                self.last_checkpoint_step = metadata.get("last_checkpoint_step", -1)
            
            print(f"📂 Loaded custom PPO policy from: {path}")
            return True
        else:
            print(f"❌ Policy file not found: {path}")
            return False

def calculate_custom_ppo_reward(generation_time, new_tokens, total_tokens, depth, top_k):
    """Calculate reward for custom PPO learning with appropriate scale"""
    # Primary reward: tokens per second (speed)
    if generation_time <= 0 or new_tokens <= 0:
        return 0.0
    
    # Reward is simply tokens per second
    tokens_per_second = new_tokens / generation_time
    return tokens_per_second
