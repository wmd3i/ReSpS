"""
PPO-based Online RL Policy for Real-time EAGLE Parameter Optimization
Uses PPO with continuous action space for more stable and efficient learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from collections import deque
import random
import json
import os
import math
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import tempfile
import shutil

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

class EAGLEParameterEnv(gym.Env):
    """Custom Gym environment for EAGLE parameter optimization"""
    
    def __init__(self, state_encoder=None):
        super(EAGLEParameterEnv, self).__init__()
        
        # Action space: continuous parameters [total_tokens, depth, top_k]
        # Normalized to [-1, 1] range, will be scaled to actual ranges
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation space: SBERT embeddings (384 dimensions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32
        )
        
        # Parameter ranges
        self.param_ranges = {
            'total_tokens': {'min': 16, 'max': 128},
            'depth': {'min': 2, 'max': 8},
            'top_k': {'min': 2, 'max': 32}
        }
        
        # State management
        self.current_context = ""
        self.state_encoder = state_encoder
        self.last_reward = 0.0
        self.episode_count = 0
        
    def reset(self):
        """Reset environment for new episode"""
        # Return a dummy observation (will be overridden when context is set)
        return np.zeros(384, dtype=np.float32)
    
    def step(self, action):
        """Execute action and return observation, reward, done, info"""
        # Convert action to parameters
        total_tokens, depth, top_k = self._action_to_parameters(action)
        
        # For now, return dummy values - actual reward will be set externally
        reward = self.last_reward
        done = True  # Each context is one episode
        info = {
            'total_tokens': total_tokens,
            'depth': depth,
            'top_k': top_k,
            'action': action.copy()
        }
        
        # Return current state as next observation
        obs = self._encode_context(self.current_context)
        
        return obs, reward, done, info
    
    def _action_to_parameters(self, action):
        """Convert normalized action [-1,1] to actual parameters"""
        # Clamp action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Scale to parameter ranges
        total_tokens = self._scale_to_range(
            action[0], 
            self.param_ranges['total_tokens']['min'], 
            self.param_ranges['total_tokens']['max']
        )
        depth = self._scale_to_range(
            action[1], 
            self.param_ranges['depth']['min'], 
            self.param_ranges['depth']['max']
        )
        top_k = self._scale_to_range(
            action[2], 
            self.param_ranges['top_k']['min'], 
            self.param_ranges['top_k']['max']
        )
        
        # Round to integers
        total_tokens = int(round(total_tokens))
        depth = int(round(depth))
        top_k = int(round(top_k))
        
        # Apply constraint: total_tokens <= top_k^(depth-1)
        max_possible_tokens = top_k ** (depth - 1)
        if total_tokens > max_possible_tokens:
            total_tokens = max_possible_tokens
        
        # Ensure minimum values
        total_tokens = max(total_tokens, self.param_ranges['total_tokens']['min'])
        depth = max(depth, self.param_ranges['depth']['min'])
        top_k = max(top_k, self.param_ranges['top_k']['min'])
        # total_tokens = min(total_tokens, top_k ** (depth - 1))
        return total_tokens, depth, top_k
    
    def _scale_to_range(self, value, min_val, max_val):
        """Scale a value from [-1, 1] to [min_val, max_val]"""
        return min_val + (value + 1) * (max_val - min_val) / 2
    
    def _encode_context(self, context):
        """Encode context using SBERT"""
        if self.state_encoder is not None:
            embedding = self.state_encoder.encode(context)
            return embedding.astype(np.float32)
        else:
            return np.zeros(384, dtype=np.float32)
    
    def set_context(self, context):
        """Set current context for state encoding"""
        self.current_context = context
        return self._encode_context(context)
    
    def set_reward(self, reward):
        """Set reward for the last action"""
        self.last_reward = reward

class WandbCallback(BaseCallback):
    """Custom callback for Wandb logging"""
    
    def __init__(self, wandb_run=None, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.wandb_run = wandb_run
        
    def _on_step(self) -> bool:
        if self.wandb_run is not None:
            # Log training metrics
            if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                wandb.log({
                    "ppo_timesteps": self.num_timesteps,
                    "ppo_learning_progress": self.num_timesteps
                })
        return True

class PPOOnlineTreePolicy:
    """PPO-based Online RL Policy for EAGLE parameter optimization"""
    
    def __init__(self, 
                 learning_rate=3e-4,
                 n_steps=2048,
                 batch_size=64,
                 n_epochs=10,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_range=0.2,
                 use_wandb=True,
                 wandb_project="eagle-ppo-rl",
                 wandb_run_name=None,
                 checkpoint_dir="checkpoints_ppo",
                 checkpoint_freq=100,
                 max_checkpoints=3,
                 verbose=1):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
                        "device": str(self.device),
                        "algorithm": "PPO",
                        "action_space": "continuous"
                    }
                )
                print(f"🔗 Wandb logging initialized: {wandb.run.get_url()}")
            else:
                print(f"🔗 Using existing wandb run: {wandb.run.get_url()}")
        else:
            print("📊 Wandb logging disabled")
        
        # Initialize SBERT for state encoding
        print("Loading SBERT model for state representation...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create custom environment
        self.env = EAGLEParameterEnv(state_encoder=self.sbert_model)
        
        # Create vectorized environment for PPO
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Initialize PPO model
        self.ppo_model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            verbose=verbose,
            device=self.device,
            tensorboard_log=None  # Disable tensorboard, use wandb instead
        )
        
        # Training counters
        self.step_count = 0
        
        # Performance tracking
        self.reward_history = []
        self.parameter_history = []
        self.tokens_per_second_history = []
        
        # Parameter ranges for reference
        self.param_ranges = self.env.param_ranges
        
        # PPO-specific tracking
        self.ppo_updates = 0
        
        print(f"PPO Online RL Policy initialized:")
        print(f"  - State dim: 384 (SBERT)")
        print(f"  - Action space: Continuous")
        print(f"  - Total tokens: {self.param_ranges['total_tokens']['min']}-{self.param_ranges['total_tokens']['max']}")
        print(f"  - Depth: {self.param_ranges['depth']['min']}-{self.param_ranges['depth']['max']}")
        print(f"  - Top-k: {self.param_ranges['top_k']['min']}-{self.param_ranges['top_k']['max']}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - PPO n_steps: {n_steps}")
        print(f"  - PPO batch_size: {batch_size}")
        print(f"  - Device: {self.device}")
        print(f"  - Wandb logging: {'enabled' if self.use_wandb else 'disabled'}")
    
    def predict_parameters(self, context, training_mode=True):
        """Predict parameters using PPO policy"""
        # Set context in environment
        obs = self.env.set_context(context)
        
        # Get action from PPO model
        action, _ = self.ppo_model.predict(obs, deterministic=not training_mode)
        
        # Convert action to parameters
        total_tokens, depth, top_k = self.env._action_to_parameters(action)
        
        # Store for learning
        if training_mode:
            self.last_obs = obs
            self.last_action = action
            self.last_params = (total_tokens, depth, top_k)
            
            mode_str = "EXPLORE" if training_mode else "EXPLOIT"
            print(f"PPO RL {mode_str}: tt={total_tokens}, d={depth}, k={top_k}")
        
        return total_tokens, depth, top_k
    
    def update_policy(self, reward, generation_time=None, new_tokens=None):
        """Update policy using PPO learning"""
        if not hasattr(self, 'last_obs'):
            return
        
        # Set reward in environment
        self.env.set_reward(reward)
        
        # Track performance
        self.reward_history.append(reward)
        self.parameter_history.append(self.last_params)
        
        if generation_time and new_tokens:
            tokens_per_sec = new_tokens / generation_time if generation_time > 0 else 0
            self.tokens_per_second_history.append(tokens_per_sec)
        
        # Accumulate experience in PPO's buffer
        # Note: PPO handles experience collection internally
        # We'll trigger learning every N steps
        self.step_count += 1
        
        # Learn from accumulated experience periodically
        if self.step_count % 32 == 0:  # Learn every 32 steps
            try:
                # Create a temporary episode for PPO to learn from
                obs = self.last_obs.reshape(1, -1)
                
                # Manually step through environment to collect experience
                self.vec_env.reset()
                self.vec_env.env_method('set_context', self.env.current_context)
                self.vec_env.env_method('set_reward', reward)
                
                # Train PPO model
                self.ppo_model.learn(total_timesteps=1, reset_num_timesteps=False)
                self.ppo_updates += 1
                
                print(f"  → PPO updated! Training step: {self.ppo_updates}, Reward: {reward:.4f}")
                
            except Exception as e:
                print(f"⚠️  PPO learning error: {e}")
        
        # Wandb logging
        if self.use_wandb:
            log_dict = {
                "step": self.step_count,
                "reward": reward,
                "ppo_updates": self.ppo_updates,
                "total_tokens": self.last_params[0],
                "depth": self.last_params[1],
                "top_k": self.last_params[2],
            }
            
            if generation_time and new_tokens:
                log_dict["tokens_per_second"] = tokens_per_sec
                log_dict["generation_time"] = generation_time
                log_dict["new_tokens"] = new_tokens
            
            if len(self.reward_history) >= 10:
                log_dict["avg_reward_10"] = np.mean(self.reward_history[-10:])
            
            wandb.log(log_dict)
        
        # Statistics
        if self.step_count % 10 == 0:
            recent_rewards = self.reward_history[-10:]
            avg_reward = np.mean(recent_rewards)
            print(f"Step {self.step_count}: Recent avg reward: {avg_reward:.3f}, PPO updates: {self.ppo_updates}")
            
            # Show parameter diversity
            recent_params = self.parameter_history[-10:]
            unique_params = len(set(recent_params))
            print(f"  → Parameter diversity: {unique_params}/10 unique combinations")
    
    def save_checkpoint(self, checkpoint_name=None):
        """Save training checkpoint"""
        if checkpoint_name is None:
            checkpoint_name = f"ppo_checkpoint_step_{self.step_count}.zip"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save PPO model
        self.ppo_model.save(checkpoint_path)
        
        # Save additional metadata
        metadata_path = checkpoint_path.replace('.zip', '_metadata.json')
        metadata = {
            'step_count': self.step_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'reward_history': self.reward_history,
            'parameter_history': self.parameter_history,
            'tokens_per_second_history': self.tokens_per_second_history,
            'param_ranges': self.param_ranges,
            'ppo_updates': self.ppo_updates,
            'policy_type': 'ppo'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"💾 PPO checkpoint saved: {checkpoint_path}")
        
        self._cleanup_old_checkpoints()
        
        if self.use_wandb and wandb.run is not None:
            try:
                artifact = wandb.Artifact(
                    name=f"ppo_checkpoint_{self.step_count}",
                    type="checkpoint",
                    description=f"PPO training checkpoint at step {self.step_count}"
                )
                artifact.add_file(checkpoint_path)
                artifact.add_file(metadata_path)
                wandb.log_artifact(artifact)
                print(f"🔗 PPO checkpoint logged to wandb")
            except Exception as e:
                print(f"⚠️  Failed to log checkpoint: {e}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load the most recent checkpoint"""
        if checkpoint_path is None:
            # Find most recent checkpoint
            if not os.path.exists(self.checkpoint_dir):
                return False
            
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                              if f.startswith('ppo_checkpoint_step_') and f.endswith('.zip')]
            
            if not checkpoint_files:
                return False
            
            def extract_step(filename):
                try:
                    return int(filename.split('_')[3].split('.')[0])
                except:
                    return 0
            
            checkpoint_files.sort(key=extract_step, reverse=True)
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_files[0])
        
        if not os.path.exists(checkpoint_path):
            return False
        
        try:
            # Load PPO model
            self.ppo_model = PPO.load(checkpoint_path, env=self.vec_env)
            
            # Load metadata
            metadata_path = checkpoint_path.replace('.zip', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.step_count = metadata.get('step_count', 0)
                self.questions_processed = metadata.get('questions_processed', 0)
                self.training_seed = metadata.get('training_seed')
                self.reward_history = metadata.get('reward_history', [])
                self.parameter_history = metadata.get('parameter_history', [])
                self.tokens_per_second_history = metadata.get('tokens_per_second_history', [])
                self.ppo_updates = metadata.get('ppo_updates', 0)
            
            print(f"✅ PPO checkpoint loaded from: {checkpoint_path}")
            print(f"  - Step count: {self.step_count}")
            print(f"  - Episodes: {len(self.reward_history)}")
            print(f"  - PPO updates: {self.ppo_updates}")
            print(f"  - Questions processed: {self.questions_processed}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load PPO checkpoint from {checkpoint_path}: {e}")
            return False
    
    def save(self, save_path):
        """Save the trained policy (interface compatibility)"""
        if save_path:
            # Save PPO model
            self.ppo_model.save(save_path)
            
            # Save metadata
            metadata_path = save_path.replace('.zip', '_metadata.json')
            metadata = {
                'step_count': self.step_count,
                'questions_processed': self.questions_processed,
                'training_seed': self.training_seed,
                'reward_history': self.reward_history,
                'parameter_history': self.parameter_history,
                'tokens_per_second_history': self.tokens_per_second_history,
                'param_ranges': self.param_ranges,
                'ppo_updates': self.ppo_updates,
                'policy_type': 'ppo'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            print(f"💾 PPO policy saved to: {save_path}")
            
            # Also log final stats
            stats = self.get_performance_stats()
            print(f"📊 Final Performance Stats:")
            print(f"  - Total episodes: {stats.get('total_episodes', 0)}")
            print(f"  - Overall avg reward: {stats.get('avg_reward_overall', 0):.3f}")
            print(f"  - Recent avg reward: {stats.get('avg_reward_recent', 0):.3f}")
            print(f"  - Parameter diversity: {stats.get('parameter_diversity', 0)}")
            print(f"  - PPO updates: {self.ppo_updates}")
        else:
            print("⚠️  No save path provided, skipping policy save")
    
    def load(self, load_path):
        """Load a trained policy (interface compatibility)"""
        if not os.path.exists(load_path):
            print(f"❌ Policy file not found: {load_path}")
            return False
        
        try:
            # Load PPO model
            self.ppo_model = PPO.load(load_path, env=self.vec_env)
            
            # Load metadata
            metadata_path = load_path.replace('.zip', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Verify this is a PPO policy
                if metadata.get('policy_type') != 'ppo':
                    print(f"❌ Policy type mismatch. Expected 'ppo', got '{metadata.get('policy_type')}'")
                    return False
                
                self.step_count = metadata.get('step_count', 0)
                self.questions_processed = metadata.get('questions_processed', 0)
                self.training_seed = metadata.get('training_seed')
                self.reward_history = metadata.get('reward_history', [])
                self.parameter_history = metadata.get('parameter_history', [])
                self.tokens_per_second_history = metadata.get('tokens_per_second_history', [])
                self.ppo_updates = metadata.get('ppo_updates', 0)
                
                # Load parameter ranges (in case they changed)
                if 'param_ranges' in metadata:
                    self.param_ranges = metadata['param_ranges']
            
            print(f"✅ PPO policy loaded from: {load_path}")
            print(f"  - Step count: {self.step_count}")
            print(f"  - Episodes: {len(self.reward_history)}")
            print(f"  - PPO updates: {self.ppo_updates}")
            print(f"  - Questions processed: {self.questions_processed}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load PPO policy from {load_path}: {e}")
            return False
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-100:]
        recent_params = self.parameter_history[-100:]
        
        param_stats = {}
        hashable_params = []
        
        for param_tuple in recent_params:
            try:
                # Ensure we have integers for each parameter
                total_tokens, depth, top_k = param_tuple
                total_tokens = int(total_tokens)
                depth = int(depth)
                top_k = int(top_k)
                
                # Create hashable tuple and key
                hashable_tuple = (total_tokens, depth, top_k)
                hashable_params.append(hashable_tuple)
                
                key = f"{total_tokens}-{depth}-{top_k}"
                param_stats[key] = param_stats.get(key, 0) + 1
            except (ValueError, TypeError) as e:
                print(f"⚠️  Skipping invalid parameter tuple {param_tuple}: {e}")
                continue
        
        return {
            'total_episodes': len(self.reward_history),
            'avg_reward_recent': np.mean(recent_rewards),
            'avg_reward_overall': np.mean(self.reward_history),
            'most_used_params': sorted(param_stats.items(), key=lambda x: x[1], reverse=True)[:5],
            'reward_trend': recent_rewards[-10:] if len(recent_rewards) >= 10 else recent_rewards,
            'parameter_diversity': len(set(hashable_params)),
            'ppo_updates': self.ppo_updates,
            'policy_type': 'ppo'
        }
    
    # Add utility methods for compatibility...
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping the most recent ones"""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith('ppo_checkpoint_step_') and f.endswith('.zip')]
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        def extract_step(filename):
            try:
                return int(filename.split('_')[3].split('.')[0])
            except:
                return 0
        
        # Sort by step number (oldest first)
        checkpoint_files.sort(key=extract_step)
        
        # Keep only the most recent max_checkpoints files
        files_to_remove = checkpoint_files[:-self.max_checkpoints]
        
        for filename in files_to_remove:
            old_path = os.path.join(self.checkpoint_dir, filename)
            metadata_path = old_path.replace('.zip', '_metadata.json')
            
            for path in [old_path, metadata_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        print(f"🗑️  Removed old checkpoint: {os.path.basename(path)}")
                except OSError as e:
                    print(f"⚠️  Failed to remove checkpoint {os.path.basename(path)}: {e}")
    
    def should_save_checkpoint(self):
        """Check if it's time to save a checkpoint"""
        return self.step_count > 0 and self.step_count % self.checkpoint_freq == 0
    
    def set_training_seed(self, seed):
        """Set training seed"""
        self.training_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"🎲 Training seed set to: {seed}")
    
    def get_resume_info(self):
        """Get resume information"""
        return {
            'step_count': self.step_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'total_episodes': len(self.reward_history),
            'ppo_updates': self.ppo_updates,
            'policy_type': 'ppo'
        }
    
    def increment_questions_processed(self, count=1):
        """Track processed questions"""
        self.questions_processed += count
        if self.should_save_checkpoint():
            self.save_checkpoint()
    
    @property
    def epsilon(self):
        """Compatibility property - PPO doesn't use epsilon-greedy exploration"""
        return 0.0  # PPO doesn't use epsilon

def calculate_ppo_online_reward(generation_time, new_tokens, total_tokens, depth, top_k):
    """
    Calculate reward for PPO online learning
    Simple reward: directly use speed (tokens/second) as reward
    """
    if generation_time <= 0 or new_tokens <= 0:
        return 0.0
    
    # Reward is simply tokens per second
    tokens_per_second = new_tokens / generation_time
    return tokens_per_second
