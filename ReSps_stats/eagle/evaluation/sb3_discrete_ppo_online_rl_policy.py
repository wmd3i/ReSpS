"""
Stable Baselines 3 Discrete PPO-based Online RL Policy for Real-time EAGLE Parameter Optimization
Uses SB3's optimized PPO implementation with discrete action space for stable learning
"""

import numpy as np
import torch
import torch.nn as nn
import os
import json
from sentence_transformers import SentenceTransformer
from collections import deque
import random
import gym
from gym import spaces

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Error: Stable Baselines 3 not available. Install with: pip install stable-baselines3")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

class EagleParameterEnv(gym.Env):
    """Custom Gym environment for EAGLE parameter optimization"""
    
    def __init__(self):
        super(EagleParameterEnv, self).__init__()
        
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
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.valid_actions))  # Only valid actions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32)  # SBERT embedding
        
        # Initialize SBERT for state encoding
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Environment state
        self.current_context = ""
        self.step_count = 0
        
        print(f"EagleParameterEnv initialized:")
        print(f"  - Total parameter combinations: {self.total_actions}")
        print(f"  - Valid parameter combinations: {len(self.valid_actions)}")
        print(f"  - Valid coverage: {len(self.valid_actions)/self.total_actions*100:.1f}%")
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
    
    def _encode_state(self, context):
        """Encode conversation context using SBERT"""
        embedding = self.sbert_model.encode(context)
        return embedding.astype(np.float32)
    
    def reset(self):
        """Reset environment state"""
        self.current_context = ""
        self.step_count = 0
        # Return zero state - will be set properly when predict_parameters is called
        return np.zeros(384, dtype=np.float32)
    
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
        
        self.step_count += 1
        
        # Return next observation (will be updated externally)
        next_obs = np.zeros(384, dtype=np.float32)
        
        return next_obs, reward, done, info

class WandbCallback(BaseCallback):
    """Custom callback for Wandb logging"""
    
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_params = []
    
    def _on_step(self) -> bool:
        # Get recent episode info if available
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                
                # Log to wandb
                if WANDB_AVAILABLE and wandb.run:
                    wandb.log({
                        'episode_reward': episode_reward,
                        'episode_length': episode_length,
                        'total_timesteps': self.num_timesteps,
                    })
        
        return True

class SB3DiscretePPOOnlineTreePolicy:
    """Stable Baselines 3 PPO-based Online RL Policy for EAGLE parameter optimization
    
    Supports both standard PPO and max-entropy PPO modes:
    - Standard PPO: Low entropy coefficient, deterministic inference
    - Max-Entropy PPO: High entropy coefficient, temperature-based inference
    """
    
    def __init__(self, 
                 learning_rate=3e-4,
                 n_steps=64,          # Reduced for online learning
                 batch_size=32,       # Smaller batch for faster updates
                 n_epochs=4,          # Reduced epochs to prevent overfitting
                 gamma=0.95,
                 gae_lambda=0.9,
                 clip_range=0.2,
                 ent_coef=0.1,        # DEFAULT: Max-Entropy PPO entropy (high)
                 vf_coef=0.5,         # Value function coefficient
                 max_grad_norm=0.5,
                 # Max-entropy specific parameters
                 enable_max_entropy=True,      # NEW: Max-entropy mode enabled by default
                 max_entropy_ent_coef=0.1,     # Higher entropy for max-entropy mode
                 inference_temperature=1.5,    # Temperature for inference-time exploration (default higher)
                 max_entropy_inference=True,   # NEW: Default enabled for max-entropy
                 use_wandb=True,
                 wandb_project="eagle-sb3-discrete-ppo",
                 wandb_run_name=None,
                 checkpoint_dir="sb3_discrete_ppo_checkpoints",
                 checkpoint_freq=100,
                 max_checkpoints=3):
        
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines 3 not available. Install with: pip install stable-baselines3")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Max-entropy RL configuration
        self.enable_max_entropy = enable_max_entropy
        self.inference_temperature = inference_temperature if enable_max_entropy else 1.0
        self.max_entropy_inference = max_entropy_inference and enable_max_entropy  # Only if max-entropy enabled
        
        # Choose entropy coefficient based on mode
        if enable_max_entropy:
            actual_ent_coef = max_entropy_ent_coef
            mode_name = "Max-Entropy PPO"
        else:
            actual_ent_coef = ent_coef
            mode_name = "Standard PPO"
        
        # Resume mechanism configuration
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        self.questions_processed = 0
        self.training_seed = None
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize wandb logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            if not wandb.run:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "learning_rate": learning_rate,
                        "n_steps": n_steps,
                        "batch_size": batch_size,
                        "n_epochs": n_epochs,
                        "gamma": gamma,
                        "gae_lambda": gae_lambda,
                        "clip_range": clip_range,
                        "ent_coef": actual_ent_coef,
                        "vf_coef": vf_coef,
                        "enable_max_entropy": enable_max_entropy,
                        "inference_temperature": self.inference_temperature,
                        "max_entropy_inference": self.max_entropy_inference,
                        "device": str(self.device),
                        "mode": mode_name
                    }
                )
                print(f"🔗 Wandb logging initialized: {wandb.run.get_url()}")
            else:
                print(f"🔗 Using existing wandb run: {wandb.run.get_url()}")
        else:
            print("📊 Wandb logging disabled")
        
        # Create custom environment
        self.env = EagleParameterEnv()
        
        # Wrap environment for SB3
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Create PPO model with determined entropy coefficient
        self.model = PPO(
            policy="MlpPolicy",
            env=self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=actual_ent_coef,    # Use mode-specific entropy coefficient
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=1,
            device=self.device,
            tensorboard_log=None  # We'll use wandb instead
        )
        
        # Training counters
        self.step_count = 0
        self.update_count = 0
        
        # Track performance with entropy metrics
        self.reward_history = []
        self.parameter_history = []
        self.tokens_per_second_history = []
        self.entropy_history = []  # Track entropy for max-entropy RL
        
        # Setup wandb callback
        self.callbacks = []
        if self.use_wandb:
            self.callbacks.append(WandbCallback())
        
        print(f"SB3 {mode_name} Policy initialized:")
        print(f"  - Mode: {mode_name}")
        print(f"  - Environment: EagleParameterEnv")
        print(f"  - Action space: {self.env.action_space.n} valid actions")
        print(f"  - Observation space: {self.env.observation_space.shape}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - PPO n_steps: {n_steps}")
        print(f"  - PPO batch_size: {batch_size}")
        print(f"  - PPO n_epochs: {n_epochs}")
        print(f"  - Entropy coefficient: {actual_ent_coef} ({'high for diversity' if enable_max_entropy else 'standard'})")
        if enable_max_entropy:
            print(f"  - Inference temperature: {self.inference_temperature}")
            print(f"  - Max-entropy inference: {self.max_entropy_inference}")
        else:
            print(f"  - Inference mode: Deterministic/Standard Stochastic")
        print(f"  - Device: {self.device}")
        print(f"  - Wandb logging: {'enabled' if self.use_wandb else 'disabled'}")
        
        if enable_max_entropy:
            print(f"🌟 MAX-ENTROPY MODE: Higher exploration, temperature-based inference")
        else:
            print(f"⚙️  STANDARD PPO MODE: Standard exploration, deterministic inference")
    
    def predict_parameters(self, context, training_mode=True):
        """Predict parameters using SB3 PPO policy (standard or max-entropy mode)"""
        # Encode state
        state = self.env._encode_state(context)
        self.env.current_context = context
        
        # Choose prediction strategy based on mode and phase
        if self.enable_max_entropy and self.max_entropy_inference and not training_mode:
            # MAX-ENTROPY INFERENCE: Use temperature-based sampling for diversity
            if self.inference_temperature != 1.0:
                # Try temperature-based sampling
                try:
                    action = self._sample_with_temperature(state, self.inference_temperature)
                    exploration_mode = "MAX-ENTROPY"
                except Exception as e:
                    print(f"⚠️  Temperature sampling failed: {e}")
                    print("   Using enhanced stochastic sampling instead")
                    action = self._enhanced_stochastic_sampling(state)
                    exploration_mode = "ENHANCED-STOCHASTIC"
            else:
                # Use stochastic prediction for exploration
                action, _ = self.model.predict(state, deterministic=False) 
                exploration_mode = "STOCHASTIC"
        elif training_mode:
            # TRAINING MODE: Always use stochastic for both modes (exploration)
            action, _ = self.model.predict(state, deterministic=False)
            exploration_mode = "EXPLORE"
        else:
            # STANDARD INFERENCE: Use deterministic prediction
            action, _ = self.model.predict(state, deterministic=True)
            exploration_mode = "DETERMINISTIC"
        
        # Convert to actual parameters
        actual_action = self.env.valid_actions[action]
        total_tokens, depth, top_k = self.env._action_to_params(actual_action)
        
        # Store for training
        if training_mode:
            self.last_state = state
            self.last_action = action
            self.last_params = (total_tokens, depth, top_k)
        
        # Debug print
        # max_tokens = top_k ** (depth - 1)
        # mode_name = "Max-Entropy PPO" if self.enable_max_entropy else "Standard PPO"
        # print(f"SB3 {mode_name} {exploration_mode}: tt={total_tokens}, d={depth}, k={top_k} (max={max_tokens})")
        
        return total_tokens, depth, top_k
    
    def _sample_with_temperature(self, state, temperature):
        """Sample action using temperature-based softmax for max-entropy exploration"""
        # Get action probabilities from the policy network
        obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Use SB3's proper prediction pipeline to get action distribution
            # This ensures proper feature extraction and preprocessing
            try:
                # Method 1: Get logits directly from the policy network
                features = self.model.policy.extract_features(obs_tensor)
                if hasattr(self.model.policy, 'mlp_extractor'):
                    # SB3 v2.x style with mlp_extractor
                    latent_pi, _ = self.model.policy.mlp_extractor(features)
                    logits = self.model.policy.action_net(latent_pi)
                else:
                    # SB3 v1.x style direct action network
                    logits = self.model.policy.action_net(features)
                
                # Apply temperature scaling for max-entropy sampling
                if temperature > 0:
                    scaled_logits = logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Sample from the distribution
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    
                    # Log entropy for monitoring
                    entropy = dist.entropy()
                    if self.use_wandb and hasattr(self, 'step_count'):
                        import wandb
                        wandb.log({
                            "inference_entropy": entropy.item(),
                            "inference_temperature": temperature,
                            "step": self.step_count
                        })
                    
                    # print(f"   Max-entropy sampling: T={temperature:.1f}, H={entropy.item():.3f}")
                    return action.item()
                else:
                    # Deterministic (argmax)
                    return torch.argmax(logits, dim=-1).item()
                    
            except Exception as e:
                print(f"⚠️  Error in direct logits extraction: {e}")
                
                # Method 2: Fallback using action probabilities from policy
                try:
                    # Get the action probabilities using the policy's forward pass
                    actions, values, log_probs = self.model.policy.forward(obs_tensor)
                    
                    # Convert log probabilities to probabilities
                    probs = torch.exp(log_probs)
                    
                    if temperature > 0 and temperature != 1.0:
                        # Apply temperature scaling
                        # Convert back to logits, scale, then to probabilities
                        logits = torch.log(probs + 1e-8)  # Add small epsilon for numerical stability
                        scaled_logits = logits / temperature
                        scaled_probs = torch.softmax(scaled_logits, dim=-1)
                        
                        # Sample from the scaled distribution
                        dist = torch.distributions.Categorical(scaled_probs)
                        action = dist.sample()
                        
                        # Log entropy for monitoring
                        entropy = dist.entropy()
                        if self.use_wandb and hasattr(self, 'step_count'):
                            import wandb
                            wandb.log({
                                "inference_entropy": entropy.item(),
                                "inference_temperature": temperature,
                                "step": self.step_count
                            })
                        
                        print(f"   Max-entropy sampling (fallback): T={temperature:.1f}, H={entropy.item():.3f}")
                        return action.item()
                    else:
                        # Use the original action from policy
                        return actions.item()
                        
                except Exception as e2:
                    print(f"⚠️  Error in fallback method: {e2}")
                    print("   Using standard stochastic PPO prediction")
                    # Final fallback to standard prediction
                    action, _ = self.model.predict(obs_tensor.cpu().numpy(), deterministic=False)
                    return action if isinstance(action, int) else action.item()
    
    def _enhanced_stochastic_sampling(self, state):
        """Enhanced stochastic sampling for max-entropy when temperature sampling fails"""
        # Sample multiple times and choose based on a strategy to increase diversity
        samples = []
        for _ in range(5):  # Take 5 samples
            action, _ = self.model.predict(state, deterministic=False)
            samples.append(action if isinstance(action, int) else action.item())
        
        # Strategy 1: Choose a random sample (increases diversity)
        chosen_action = random.choice(samples)
        
        # Log diversity metrics
        unique_samples = len(set(samples))
        if self.use_wandb and hasattr(self, 'step_count'):
            import wandb
            wandb.log({
                "enhanced_sampling_diversity": unique_samples / 5.0,
                "enhanced_sampling_unique_actions": unique_samples,
                "step": self.step_count
            })
        
        print(f"   Enhanced stochastic: {unique_samples}/5 unique actions")
        return chosen_action
    
    def update_policy(self, reward, generation_time=None, new_tokens=None):
        """Update policy with reward from last action"""
        if not hasattr(self, 'last_state'):
            return
        
        # Create a temporary episode for the single step
        obs = self.vec_env.reset()
        obs[0] = self.last_state
        
        # Take action and get environment response
        obs, _, done, info = self.vec_env.step([self.last_action])
        
        # Manually set the reward in the environment
        # Since SB3 doesn't directly support external rewards, we'll use a different approach
        
        # Track performance
        self.reward_history.append(reward)
        self.parameter_history.append(self.last_params)
        
        # Track tokens per second if provided
        if generation_time and new_tokens:
            tps = new_tokens / generation_time
            self.tokens_per_second_history.append(tps)
        
        # print(f"  → Reward: {reward:.3f} for {self.last_params}")
        
        # Update model every few steps
        if len(self.reward_history) % 32 == 0:  # Update every 32 steps
            # Create training data from recent experiences
            # Note: This is a simplified approach. For full SB3 integration, 
            # you'd want to use a custom environment with proper episode handling
            
            self.update_count += 1
            # print(f"SB3 PPO Update #{self.update_count}: Manual reward integration")
        
        # Increment step counter
        self.step_count += 1
        
        # Save checkpoint periodically
        if self.should_save_checkpoint():
            self.save_checkpoint()
        
        # Progress logging
        if self.step_count % 10 == 0:
            recent_rewards = self.reward_history[-10:] if len(self.reward_history) >= 10 else self.reward_history
            avg_recent_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            
            # print(f"Online RL Update: Reward={reward:.3f}, Avg Recent Reward={avg_recent_reward:.3f}, Tokens/sec={tps:.1f}" if 'tps' in locals() else f"Online RL Update: Reward={reward:.3f}, Avg Recent Reward={avg_recent_reward:.3f}")
            # print(f"📊 Progress: {self.questions_processed}/{400 if hasattr(self, 'total_questions') else '?'} questions, Step: {self.step_count}, SB3 Updates: {self.update_count}")
        
        # Wandb logging
        if self.use_wandb:
            log_data = {
                "reward": reward,
                "total_tokens": self.last_params[0],
                "depth": self.last_params[1], 
                "top_k": self.last_params[2],
                "step": self.step_count,
                "questions_processed": self.questions_processed,
                "mode": "Max-Entropy PPO" if self.enable_max_entropy else "Standard PPO",
                # "exploration_mode": "MAX-ENTROPY" if self.max_entropy_inference and not training_mode else "DETERMINISTIC" if not self.enable_max_entropy else "STOCHASTIC",
                "tokens_per_second": self.tokens_per_second_history[-1] if self.tokens_per_second_history else 0.0,
                "entropy": self.entropy_history[-1] if self.entropy_history else 0.0,
                "update_count": self.update_count,
            }
            
            # Add tokens per second if available
            if 'tps' in locals():
                log_data["tokens_per_second"] = tps
            
            # Add averaging windows
            if len(self.reward_history) >= 10:
                log_data["avg_reward_10"] = torch.mean(torch.tensor(self.reward_history[-10:])).item()
                log_data["avg_tokens_per_second_10"] = torch.mean(torch.tensor(self.tokens_per_second_history[-10:])).item() if self.tokens_per_second_history else 0.0
            if len(self.reward_history) >= 50:
                log_data["avg_reward_50"] = torch.mean(torch.tensor(self.reward_history[-50:])).item()
                log_data["avg_tokens_per_second_50"] = torch.mean(torch.tensor(self.tokens_per_second_history[-50:])).item() if self.tokens_per_second_history else 0.0

            wandb.log(log_data)
    
    def save_checkpoint(self, checkpoint_name=None):
        """Save training checkpoint"""
        if checkpoint_name is None:
            checkpoint_name = f"sb3_discrete_ppo_checkpoint_step_{self.step_count}"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save SB3 model
        self.model.save(checkpoint_path)
        
        # Save additional metadata - convert tensors to JSON-serializable types
        def convert_to_serializable(obj):
            """Convert tensors and other objects to JSON-serializable types"""
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        metadata = {
            'step_count': self.step_count,
            'update_count': self.update_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'reward_history': convert_to_serializable(self.reward_history),
            'parameter_history': convert_to_serializable(self.parameter_history),
            'tokens_per_second_history': convert_to_serializable(self.tokens_per_second_history),
            'entropy_history': convert_to_serializable(self.entropy_history),
        }
        
        metadata_path = checkpoint_path + "_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"💾 SB3 Discrete PPO checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load training checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path + ".zip"):
            print(f"❌ No checkpoint found to load")
            return False
        
        try:
            # Load SB3 model
            self.model = PPO.load(checkpoint_path, env=self.vec_env, device=self.device)
            
            # Load metadata
            metadata_path = checkpoint_path + "_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.step_count = metadata['step_count']
                self.update_count = metadata['update_count']
                self.questions_processed = metadata.get('questions_processed', 0)
                self.training_seed = metadata.get('training_seed')
                self.reward_history = metadata['reward_history']
                self.parameter_history = metadata['parameter_history']
                self.tokens_per_second_history = metadata['tokens_per_second_history']
                self.entropy_history = metadata.get('entropy_history', [])
            
            print(f"✅ SB3 Discrete PPO checkpoint loaded: {checkpoint_path}")
            print(f"   Resuming from step {self.step_count}, update {self.update_count}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return False
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith('sb3_discrete_ppo_checkpoint_step_') and f.endswith('.zip')]
        if not checkpoint_files:
            return None
        
        # Sort by step number to get latest
        def extract_step(filename):
            try:
                return int(filename.replace('sb3_discrete_ppo_checkpoint_step_', '').replace('.zip', ''))
            except:
                return 0
        
        latest_file = max(checkpoint_files, key=extract_step)
        return os.path.join(self.checkpoint_dir, latest_file.replace('.zip', ''))
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to keep only max_checkpoints files"""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith('sb3_discrete_ppo_checkpoint_step_') and f.endswith('.zip')]
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by step number
        def extract_step(filename):
            try:
                return int(filename.replace('sb3_discrete_ppo_checkpoint_step_', '').replace('.zip', ''))
            except:
                return 0
        
        checkpoint_files.sort(key=extract_step)
        
        # Remove oldest checkpoints
        files_to_remove = checkpoint_files[:-self.max_checkpoints]
        for file_to_remove in files_to_remove:
            file_path = os.path.join(self.checkpoint_dir, file_to_remove)
            metadata_path = file_path.replace('.zip', '_metadata.json')
            
            try:
                os.remove(file_path)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                print(f"🗑️ Removed old checkpoint: {file_to_remove}")
            except OSError as e:
                print(f"⚠️ Failed to remove checkpoint: {e}")
    
    def should_save_checkpoint(self):
        """Check if it's time to save a checkpoint"""
        return self.step_count % self.checkpoint_freq == 0
    
    def set_training_seed(self, seed):
        """Set training seed for reproducible shuffling"""
        self.training_seed = seed
    
    def get_resume_info(self):
        """Get information needed for resume"""
        return {
            'step_count': self.step_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'ppo_updates': self.update_count,
            'total_episodes': len(self.reward_history)
        }
    
    def increment_questions_processed(self, count=1):
        """Increment questions processed counter"""
        self.questions_processed += count
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-100:]  # Last 100 rewards
        stats = {
            'total_steps': self.step_count,
            'total_updates': self.update_count,
            'avg_reward': torch.mean(torch.tensor(self.reward_history)).item(),
            'recent_avg_reward': torch.mean(torch.tensor(recent_rewards)).item(),
            'best_reward': torch.max(torch.tensor(self.reward_history)).item(),
            'questions_processed': self.questions_processed
        }
        
        if self.tokens_per_second_history:
            recent_tps = self.tokens_per_second_history[-100:]
            stats.update({
                'avg_tokens_per_second': torch.mean(torch.tensor(self.tokens_per_second_history)).item(),
                'recent_avg_tokens_per_second': torch.mean(torch.tensor(recent_tps)).item(),
                'best_tokens_per_second': torch.max(torch.tensor(self.tokens_per_second_history)).item()
            })
        
        return stats
    
    def save(self, path):
        """Save trained policy (final model)"""
        # Remove .pth extension if present and add SB3 naming
        if path.endswith('.pth'):
            path = path.replace('.pth', '_sb3')
        
        # Save SB3 model
        self.model.save(path)
        
        # Save performance stats
        stats = self.get_performance_stats()
        stats_path = path + "_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"💾 SB3 Discrete PPO policy saved to: {path}.zip")
        print(f"💾 Performance stats saved to: {stats_path}")
    
    def load(self, path):
        """Load trained policy"""
        # Handle .pth extension
        if path.endswith('.pth'):
            path = path.replace('.pth', '_sb3')
        
        if not os.path.exists(path + ".zip"):
            print(f"❌ Policy file not found: {path}.zip")
            return False
        
        try:
            self.model = PPO.load(path, env=self.vec_env, device=self.device)
            print(f"✅ SB3 Discrete PPO policy loaded from: {path}.zip")
            return True
        except Exception as e:
            print(f"❌ Error loading policy: {e}")
            return False


def calculate_sb3_discrete_ppo_reward(generation_time, new_tokens, total_tokens, depth, top_k):
    """Calculate reward for SB3 discrete PPO learning with appropriate scale"""
    # Primary reward: tokens per second (speed)
    if generation_time <= 0 or new_tokens <= 0:
        return 0.0
    
    # Reward is simply tokens per second
    tokens_per_second = new_tokens / generation_time
    return tokens_per_second