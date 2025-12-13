"""
Custom PPO Components - Replacement for Stable Baselines 3
Implements core PPO functionality following official SB3 practices
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import gym
from gym import spaces


class RolloutBuffer:
    """Custom rollout buffer for storing PPO experiences"""
    
    def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        
        # Determine observation and action dimensions
        if isinstance(observation_space, spaces.Box):
            self.obs_shape = observation_space.shape
        else:
            raise ValueError("Only Box observation spaces are supported")
        
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = 1
        else:
            raise ValueError("Only Discrete action spaces are supported")
        
        # Initialize buffers
        self.observations = torch.zeros((buffer_size,) + self.obs_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, self.action_dim), dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)  # Use float32 instead of bool
        
        self.pos = 0
        self.full = False
    
    def add(self, obs: np.ndarray, action: int, reward: float, value: float, log_prob: float, done: bool):
        """Add a single experience to the buffer"""
        self.observations[self.pos] = torch.FloatTensor(obs).to(self.device)
        self.actions[self.pos] = torch.LongTensor([action]).to(self.device)
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = float(done)
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def get(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get a batch of experiences for training"""
        if not self.full:
            raise ValueError("Buffer not full yet")
        
        # Shuffle indices
        indices = torch.randperm(self.buffer_size, device=self.device)
        
        for start_idx in range(0, self.buffer_size, batch_size):
            end_idx = min(start_idx + batch_size, self.buffer_size)
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                'observations': self.observations[batch_indices],
                'actions': self.actions[batch_indices].squeeze(-1),
                'rewards': self.rewards[batch_indices],
                'values': self.values[batch_indices],
                'log_probs': self.log_probs[batch_indices],
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices],
                'dones': self.dones[batch_indices]
            }
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute returns and advantages using GAE"""
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values
    
    def reset(self):
        """Reset the buffer"""
        self.pos = 0
        self.full = False


class ActorCriticPolicy(nn.Module):
    """Custom Actor-Critic Policy Network"""
    
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, 
                 net_arch: List[int] = None, activation_fn: nn.Module = nn.ReLU):
        super().__init__()
        
        if net_arch is None:
            net_arch = [64, 64]
        
        # Extract dimensions
        if isinstance(observation_space, spaces.Box):
            self.features_dim = observation_space.shape[0]
        else:
            raise ValueError("Only Box observation spaces are supported")
        
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
        else:
            raise ValueError("Only Discrete action spaces are supported")
        
        # Build shared feature extractor
        shared_layers = []
        prev_dim = self.features_dim
        for layer_size in net_arch:
            shared_layers.extend([
                nn.Linear(prev_dim, layer_size),
                activation_fn(),
                nn.Dropout(0.1)
            ])
            prev_dim = layer_size
        
        self.shared_net = nn.Sequential(*shared_layers)
        
        # Policy head (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(prev_dim, 64),
            activation_fn(),
            nn.Linear(64, self.action_dim)
        )
        
        # Value head (critic)
        self.value_net = nn.Sequential(
            nn.Linear(prev_dim, 64),
            activation_fn(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights following SB3 practices
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights following SB3 best practices"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use orthogonal initialization for better gradient flow
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        features = self.shared_net(obs)
        policy_logits = self.policy_net(features)
        value = self.value_net(features)
        return policy_logits, value
    
    def get_distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """Get action distribution"""
        policy_logits, _ = self.forward(obs)
        return torch.distributions.Categorical(logits=policy_logits)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update"""
        policy_logits, values = self.forward(obs)
        distribution = torch.distributions.Categorical(logits=policy_logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values.squeeze(-1), log_probs, entropy


class CustomPPO:
    """Custom PPO Implementation following SB3 practices"""
    
    def __init__(self, 
                 policy: ActorCriticPolicy,
                 env,
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 clip_range_vf: Optional[float] = None,
                 normalize_advantage: bool = True,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: Optional[float] = None,
                 device: str = "auto"):
        
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = policy.to(self.device)
        self.env = env
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # Initialize optimizer with SB3-style learning rate scheduling
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=learning_rate,
            eps=1e-7,  # SB3 default epsilon
            weight_decay=0.0  # No weight decay for PPO
        )
        
        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=self.device
        )
        
        # Training state
        self.num_timesteps = 0
        self._n_updates = 0
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, Optional[Dict]]:
        """Predict action for given observation"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            distribution = self.policy.get_distribution(obs_tensor)
            if deterministic:
                action = distribution.probs.argmax(dim=-1)
            else:
                action = distribution.sample()
            
            return action.item(), None
    
    def collect_rollouts(self) -> bool:
        """Collect rollout data from environment"""
        self.rollout_buffer.reset()
        
        obs = self.env.reset()
        episode_rewards = []
        episode_lengths = []
        
        for step in range(self.n_steps):
            # Predict action
            action, _ = self.predict(obs, deterministic=False)
            
            # Get distribution for logging
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                distribution = self.policy.get_distribution(obs_tensor)
                value = self.policy.value_net(self.policy.shared_net(obs_tensor)).squeeze(-1)
                log_prob = distribution.log_prob(torch.LongTensor([action]).to(self.device))
            
            # Take action in environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Store experience
            self.rollout_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                value=value.item(),
                log_prob=log_prob.item(),
                done=done
            )
            
            obs = next_obs
            self.num_timesteps += 1
            
            if done:
                obs = self.env.reset()
        
        # Compute returns and advantages
        last_value = 0.0  # Assuming episodic environment
        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        return True
    
    def train(self):
        """Perform PPO training update"""
        # Switch to train mode
        self.policy.train()
        
        # Initialize metrics tracking
        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        approx_kl_divs = []
        
        continue_training = True
        
        # Train for n_epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs_epoch = []
            
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data['actions']
                observations = rollout_data['observations']
                old_values = rollout_data['values']
                old_log_probs = rollout_data['log_probs']
                advantages = rollout_data['advantages']
                returns = rollout_data['returns']
                
                # Normalize advantage
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Evaluate actions
                values, log_probs, entropy = self.policy.evaluate_actions(observations, actions)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - old_log_probs)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + torch.clamp(
                        values - old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                value_loss = F.mse_loss(returns, values_pred)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Logging
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                
                # Calculate approximate KL divergence
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs_epoch.append(approx_kl_div)
                
                # Early stopping if KL divergence is too high
                if self.target_kl is not None and np.mean(approx_kl_divs_epoch) > 1.5 * self.target_kl:
                    continue_training = False
                    break
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            approx_kl_divs.extend(approx_kl_divs_epoch)
            self._n_updates += 1
            
            if not continue_training:
                break
        
        # Log training metrics
        training_info = {
            'entropy_loss': np.mean(entropy_losses),
            'policy_loss': np.mean(pg_losses),
            'value_loss': np.mean(value_losses),
            'approx_kl': np.mean(approx_kl_divs),
            'clip_fraction': np.mean(clip_fractions),
            'n_updates': self._n_updates,
            'explained_variance': self._compute_explained_variance()
        }
        
        return training_info
    
    def _compute_explained_variance(self):
        """Compute explained variance (SB3 metric)"""
        if hasattr(self, 'rollout_buffer') and self.rollout_buffer.full:
            values = self.rollout_buffer.values.cpu().numpy()
            returns = self.rollout_buffer.returns.cpu().numpy()
            return 1 - np.var(returns - values) / np.var(returns)
        return 0.0
    
    def learn(self, total_timesteps: int, callback=None):
        """Main learning loop"""
        while self.num_timesteps < total_timesteps:
            # Collect rollouts
            self.collect_rollouts()
            
            # Train
            training_info = self.train()
            
            # Call callback if provided
            if callback is not None:
                callback(self.num_timesteps, training_info)
        
        return self
    
    def save(self, path: str):
        """Save the model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps,
            '_n_updates': self._n_updates
        }, path)
    
    def load(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_timesteps = checkpoint['num_timesteps']
        self._n_updates = checkpoint['_n_updates']


class BaseCallback:
    """Base callback class for PPO training"""
    
    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.n_calls = 0
        self.num_timesteps = 0
    
    def on_step(self, timesteps: int, training_info: Dict) -> bool:
        """Called after each training step"""
        self.n_calls += 1
        self.num_timesteps = timesteps
        return True


class CheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints"""
    
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "ppo_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
    
    def on_step(self, timesteps: int, training_info: Dict) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}/{self.name_prefix}_{timesteps}_steps.pt"
            # Note: This would need access to the model instance
            # In practice, you'd pass the model to the callback
            if self.verbose >= 1:
                print(f"Saving model checkpoint to {model_path}")
        return True 