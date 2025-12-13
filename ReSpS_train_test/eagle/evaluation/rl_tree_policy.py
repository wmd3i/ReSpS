"""
RL agent for tree generation parameter optimization.
This module contains the RL environment and utilities for training
and using a policy to optimize total_tokens, depth, and top_k parameters.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sentence_transformers import SentenceTransformer
from stable_baselines3 import PPO
import time
import torch


def calculate_reward(generation_time, acceptance_rate, num_accepted_tokens, 
                     total_tokens, depth, top_k):
    """
    Calculate reward based on generation performance metrics.
    
    Args:
        generation_time: Time taken for generation
        acceptance_rate: Ratio of accepted tokens to total tokens
        num_accepted_tokens: Number of tokens actually accepted
        total_tokens: Total number of tokens in the tree
        depth: Depth of the speculation tree
        top_k: Top-k parameter for tree generation
    
    Returns:
        total_reward: Combined reward score
    """
    speed = num_accepted_tokens / generation_time if generation_time > 0 else 0
    speed_reward = np.log(speed + 1) * 2.0
    acceptance_reward = acceptance_rate * 3.0
    efficiency_penalty = (total_tokens + depth + top_k) * 0.01
    quality_bonus = 2.0 if acceptance_rate > 0.8 and speed > 10 else 0.0
    total_reward = speed_reward + acceptance_reward - efficiency_penalty + quality_bonus
    return total_reward


class TreeRLEnvironment(gym.Env):
    """
    RL Environment for optimizing EAGLE tree generation parameters.
    
    State: Text embedding from SBERT
    Action: Discrete choices for total_tokens, depth, and top_k
    Reward: Based on generation speed, acceptance rate, and efficiency
    """
    
    def __init__(self, questions):
        super().__init__()
        self.questions = questions
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        self.current_idx = 0
        
        # Dynamic parameter bins for RL optimization
        # Moderate ranges that balance exploration with stability
        self.total_token_bins = [55, 60, 65, 70]  # Moderate range around default
        self.depth_bins = [4, 5, 6]  # Allow some depth variation
        self.top_k_bins = [8, 10, 12]  # Moderate top_k variation
        
        self.action_space = spaces.MultiDiscrete([
            len(self.total_token_bins),
            len(self.depth_bins),
            len(self.top_k_bins)
        ])
        
        # Observation space is SBERT embedding (384 dimensions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment to start of question sequence."""
        super().reset(seed=seed)
        self.current_idx = 0
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def _get_obs(self):
        """Get current observation (SBERT embedding of current question)."""
        if self.current_idx >= len(self.questions):
            return np.zeros(384, dtype=np.float32)
        
        text = " ".join(self.questions[self.current_idx]["turns"])
        emb = self.sbert.encode([text], convert_to_numpy=True)[0]
        return emb.astype(np.float32)
    
    def step(self, action):
        """
        Take action and return next state, reward, done flag, and info.
        
        Note: This is a simulated step for training. In actual usage,
        the reward will be calculated from real generation metrics.
        """
        total_tokens = self.total_token_bins[action[0]]
        depth = self.depth_bins[action[1]]
        top_k = self.top_k_bins[action[2]]
        
        # Simulate generation metrics for training
        # In real usage, these will come from actual generation
        generation_time = np.random.uniform(0.8, 2.5)
        num_accepted_tokens = np.random.randint(1, total_tokens)
        acceptance_rate = num_accepted_tokens / total_tokens
        
        reward = calculate_reward(
            generation_time, acceptance_rate, num_accepted_tokens,
            total_tokens, depth, top_k
        )
        
        self.current_idx += 1
        done = (self.current_idx >= len(self.questions))
        truncated = False  # Add truncated flag for gymnasium compatibility
        obs = self._get_obs()
        
        info = {
            "total_tokens": total_tokens,
            "depth": depth,
            "top_k": top_k,
            "generation_time": generation_time,
            "acceptance_rate": acceptance_rate,
            "num_accepted_tokens": num_accepted_tokens
        }
        
        return obs, reward, done, truncated, info


def train_and_save_policy(questions, model_path="ppo_tree_policy_discrete.zip", 
                         total_timesteps=1000):
    """
    Train RL policy and save to disk.
    
    Args:
        questions: List of questions for training
        model_path: Path to save trained model
        total_timesteps: Number of training timesteps
    """
    env = TreeRLEnvironment(questions)
    model = PPO("MlpPolicy", env, verbose=1, batch_size=8, n_steps=16)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"Policy saved to {model_path}")


class RLTreePolicy:
    """
    Wrapper for the trained RL policy that predicts tree parameters.
    """
    
    def __init__(self, policy_path="ppo_tree_policy_discrete.zip"):
        """
        Initialize RL policy and SBERT encoder.
        
        Args:
            policy_path: Path to trained PPO policy
        """
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        self.policy = PPO.load(policy_path)
        
        # Parameter bins matching the training environment
        self.total_token_bins = [32, 48, 64, 80, 96]
        self.depth_bins = [2, 4, 6, 8, 10]
        self.top_k_bins = [4, 8, 12, 16, 20]
    
    def predict_parameters(self, text, deterministic=True):
        """
        Predict tree generation parameters for given text.
        
        Args:
            text: Input text to encode and predict parameters for
            deterministic: Whether to use deterministic policy prediction
        
        Returns:
            tuple: (total_tokens, depth, top_k)
        """
        # Encode text to get state representation
        embedding = self.sbert.encode([text], convert_to_numpy=True)[0].astype(np.float32)
        
        # Predict action using trained policy
        action, _ = self.policy.predict(embedding, deterministic=deterministic)
        
        # Map discrete actions to parameter values
        total_tokens = self.total_token_bins[action[0]]
        depth = self.depth_bins[action[1]]
        top_k = self.top_k_bins[action[2]]
        
        # Validate and clamp parameters to extremely conservative ranges compatible with EAGLE model
        # Staying very close to defaults: total_token=60, depth=5, top_k=10
        total_tokens = max(50, min(65, total_tokens))
        depth = max(4, min(5, depth))  # Keep depth very close to default 5
        top_k = max(8, min(10, top_k))  # Keep top_k close to default 10
        
        return total_tokens, depth, top_k


def calculate_real_reward(generation_time, new_tokens, total_tokens, depth, top_k):
    """
    Calculate reward from real generation metrics.
    This is used during actual inference to provide feedback.
    
    Args:
        generation_time: Actual time taken for generation
        new_tokens: Number of new tokens generated
        total_tokens: Total tokens parameter used
        depth: Depth parameter used
        top_k: Top-k parameter used
    
    Returns:
        reward: Calculated reward value
    """
    if generation_time <= 0:
        return 0.0
    
    speed = new_tokens / generation_time
    speed_reward = np.log(speed + 1) * 2.0
    
    # Efficiency penalty for using larger parameters
    efficiency_penalty = (total_tokens + depth + top_k) * 0.01
    
    # Quality bonus for good speed
    quality_bonus = 2.0 if speed > 10 else 0.0
    
    total_reward = speed_reward - efficiency_penalty + quality_bonus
    return total_reward
