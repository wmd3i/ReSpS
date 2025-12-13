"""
Discrete PPO-based Online RL Policy for Real-time EAGLE Parameter Optimization
Uses PPO with discrete action space for stable learning with parameter bins
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from sentence_transformers import SentenceTransformer
from collections import deque
import random
import json
import os
import math

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for discrete PPO with shared backbone"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(), 
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Special initialization for actor output layer (smaller values for exploration)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
    
    def forward(self, state):
        """Forward pass through both actor and critic"""
        features = self.backbone(state)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value
    
    def get_action_and_value(self, state, action=None):
        """Get action distribution and value, optionally evaluate specific action"""
        action_logits, value = self.forward(state)
        
        # Create categorical distribution
        action_dist = Categorical(logits=action_logits)
        
        if action is None:
            # Sample action
            action = action_dist.sample()
        
        return action, action_dist.log_prob(action), action_dist.entropy(), value

class DiscretePPOOnlineTreePolicy:
    """PPO-based Online RL Policy with discrete action space for EAGLE parameter optimization"""
    
    def __init__(self, 
                 learning_rate=1e-4,  # Reduced from 3e-4 for more stable learning
                 ppo_epochs=4,        # Reduced from 10 to prevent overfitting
                 batch_size=32,       # Reduced from 64 for faster updates
                 clip_range=0.2,
                 value_coeff=0.5,
                 entropy_coeff=0.05,  # Increased from 0.01 for more exploration
                 max_grad_norm=0.5,
                 gamma=0.95,          # Reduced from 0.99 for faster learning
                 gae_lambda=0.9,      # Reduced from 0.95 for faster learning
                 use_wandb=True,
                 wandb_project="eagle-discrete-ppo",
                 wandb_run_name=None,
                 checkpoint_dir="discrete_ppo_checkpoints",
                 checkpoint_freq=100,
                 max_checkpoints=3):
        
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
                        "ppo_epochs": ppo_epochs,
                        "batch_size": batch_size,
                        "clip_range": clip_range,
                        "value_coeff": value_coeff,
                        "entropy_coeff": entropy_coeff,
                        "gamma": gamma,
                        "gae_lambda": gae_lambda,
                        "device": str(self.device)
                    }
                )
                print(f"🔗 Wandb logging initialized: {wandb.run.get_url()}")
            else:
                print(f"🔗 Using existing wandb run: {wandb.run.get_url()}")
        else:
            print("📊 Wandb logging disabled")
        
        # Parameter bins for 5 actions
        self.total_tokens_bins = [32, 48, 64, 80, 96]  # 5 options
        self.depth_bins = [3, 4, 5, 6, 7]  # 5 options  
        self.top_k_bins = [4, 8, 12, 16, 20]  # 5 options
        ### Best combination ranges
        self.top_k_bins = [8, 12, 16, 20, 32]
        self.depth_bins = [3, 4, 5, 6, 7, 8]
        self.total_tokens_bins = [32, 48, 64, 80, 96, 128]
        # Action space dimensions
        self.n_total_tokens = len(self.total_tokens_bins)
        self.n_depth = len(self.depth_bins)
        self.n_top_k = len(self.top_k_bins)
        self.total_actions = self.n_total_tokens * self.n_depth * self.n_top_k
        
        # Precompute valid actions to avoid constraint violations
        self.valid_actions = self._precompute_valid_actions()
        print(f"Action space: {self.total_actions} total, {len(self.valid_actions)} valid")
        
        # Initialize SBERT for state encoding
        print("Loading SBERT model for state representation...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.state_dim = 384  # SBERT embedding dimension
        
        # Initialize Actor-Critic network
        self.network = ActorCriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.total_actions
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        
        # PPO hyperparameters
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Experience buffer for PPO updates
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Training counters
        self.step_count = 0
        self.update_count = 0
        
        # Performance tracking
        self.reward_history = []
        self.parameter_history = []
        self.loss_history = []
        self.tokens_per_second_history = []
        
        # Wandb logging setup
        if self.use_wandb:
            wandb.config.update({
                "total_actions": self.total_actions,
                "valid_actions": len(self.valid_actions),
                "valid_coverage": len(self.valid_actions)/self.total_actions,
                "total_tokens_bins": self.total_tokens_bins,
                "depth_bins": self.depth_bins,
                "top_k_bins": self.top_k_bins
            })
        
        print(f"Discrete PPO Policy initialized:")
        print(f"  - State dim: {self.state_dim}")
        print(f"  - Action space: {self.total_actions} total ({self.n_total_tokens}×{self.n_depth}×{self.n_top_k})")
        print(f"  - Valid actions: {len(self.valid_actions)} ({len(self.valid_actions)/self.total_actions*100:.1f}%)")
        print(f"  - Constraint: total_tokens ≤ top_k^(depth-1)")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - PPO epochs: {ppo_epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Device: {self.device}")
        print(f"  - Wandb logging: {'enabled' if self.use_wandb else 'disabled'}")
    
    def _encode_state(self, context):
        """Encode conversation context using SBERT"""
        embedding = self.sbert_model.encode(context)
        return torch.FloatTensor(embedding).to(self.device)
    
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
    
    def _precompute_valid_actions(self):
        """Precompute valid action indices based on constraint: total_tokens <= top_k^(depth-1)"""
        valid_actions = []
        for action in range(self.total_actions):
            total_tokens, depth, top_k = self._action_to_params(action)
            if self._is_valid_combination(total_tokens, depth, top_k):
                valid_actions.append(action)
        return valid_actions
    
    def _is_valid_combination(self, total_tokens, depth, top_k):
        """Check if parameter combination satisfies constraint"""
        max_tokens = top_k ** (depth - 1)
        return total_tokens <= max_tokens
    
    def predict_parameters(self, context, training_mode=True):
        """Predict parameters using PPO policy with improved exploration"""
        state = self._encode_state(context)
        
        with torch.no_grad():
            # Get action logits and state value
            action_logits, state_value = self.network(state.unsqueeze(0))
            
            # Mask invalid actions by setting their logits to very negative values
            masked_logits = action_logits.clone()
            valid_mask = torch.zeros(self.total_actions, dtype=torch.bool)
            
            for valid_action in self.valid_actions:
                valid_mask[valid_action] = True
            
            # Set invalid actions to very negative values
            masked_logits[0, ~valid_mask] = -1e8
            
            # Create distribution with masked logits
            masked_dist = Categorical(logits=masked_logits)
            
            if training_mode:
                # Sample from masked distribution (exploration)
                action = masked_dist.sample()
                log_prob = masked_dist.log_prob(action)
                
                # Add exploration noise for better parameter space coverage
                if self.step_count < 100:  # More exploration in early training
                    if torch.rand(1).item() < 0.3:  # 30% random exploration initially
                        # Select random valid action and create tensor on correct device
                        random_valid_idx = torch.randint(0, len(self.valid_actions), (1,), device=self.device)
                        action = torch.tensor([self.valid_actions[random_valid_idx.item()]], device=self.device)
                        log_prob = masked_dist.log_prob(action)
            else:
                # Use greedy policy (highest probability valid action) for inference
                action = masked_logits.argmax(dim=1)
                log_prob = masked_dist.log_prob(action)
        
        # Convert to parameters
        action_item = action.item()
        total_tokens, depth, top_k = self._action_to_params(action_item)
        
        # Validate the action is actually valid (safety check)
        if not self._is_valid_combination(total_tokens, depth, top_k):
            print(f"⚠️ Invalid action selected: {action_item} -> ({total_tokens}, {depth}, {top_k})")
            # Fall back to a safe default
            action_item = self.valid_actions[0]
            total_tokens, depth, top_k = self._action_to_params(action_item)
        
        # Store for training
        if training_mode:
            self.last_state = state
            self.last_action = action_item
            self.last_log_prob = log_prob.item()
            self.last_value = state_value.item()
            self.last_params = (total_tokens, depth, top_k)
            
            # Debug print
            max_tokens = top_k ** (depth - 1)
            exploration_status = "EXPLORE" if self.step_count < 100 and torch.rand(1).item() < 0.3 else "POLICY"
            print(f"Discrete PPO {exploration_status}: tt={total_tokens}, d={depth}, k={top_k} (max={max_tokens})")
        else:
            # Inference mode
            max_tokens = top_k ** (depth - 1)
            print(f"Discrete PPO INFERENCE: tt={total_tokens}, d={depth}, k={top_k} (max={max_tokens})")
        
        return total_tokens, depth, top_k
    
    def update_policy(self, reward, generation_time=None, new_tokens=None):
        """Store experience and update policy using PPO with continuous learning"""
        if not hasattr(self, 'last_state'):
            return
        
        # Store experience in buffer
        self.states.append(self.last_state.cpu())
        self.actions.append(self.last_action)
        self.rewards.append(reward)
        self.log_probs.append(self.last_log_prob)
        self.values.append(self.last_value)
        self.dones.append(False)  # Changed: Treat as continuous episodes for better learning
        
        # Track performance
        self.reward_history.append(reward)
        self.parameter_history.append(self.last_params)
        
        # Track tokens per second if provided
        if generation_time and new_tokens:
            tps = new_tokens / generation_time
            self.tokens_per_second_history.append(tps)
        
        print(f"  → Reward: {reward:.3f} for {self.last_params}")
        
        # More frequent PPO updates for better learning (smaller batches)
        min_batch_size = max(8, self.batch_size // 4)  # Allow smaller batches
        if len(self.states) >= min_batch_size:
            self._ppo_update()
        
        # Increment step counter
        self.step_count += 1
        
        # Save checkpoint periodically
        if self.should_save_checkpoint():
            self.save_checkpoint()
        
        # Enhanced wandb logging with multiple averaging windows
        recent_rewards = self.reward_history[-10:] if len(self.reward_history) >= 10 else self.reward_history
        avg_recent_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        
        # Progress logging every 10 steps
        if self.step_count % 10 == 0:
            print(f"Online RL Update: Reward={reward:.3f}, Avg Recent Reward={avg_recent_reward:.3f}, Tokens/sec={tps:.1f}" if 'tps' in locals() else f"Online RL Update: Reward={reward:.3f}, Avg Recent Reward={avg_recent_reward:.3f}")
            print(f"📊 Progress: {self.questions_processed}/{400 if hasattr(self, 'total_questions') else '?'} questions, Step: {self.step_count}, PPO Updates: {self.update_count}")
        
        # Wandb logging
        if self.use_wandb:
            log_data = {
                "reward": reward,
                "avg_recent_reward": avg_recent_reward,
                "total_tokens": self.last_params[0],
                "depth": self.last_params[1], 
                "top_k": self.last_params[2],
                "step": self.step_count,
                "questions_processed": self.questions_processed
            }
            
            # Add tokens per second if available
            if 'tps' in locals():
                log_data["tokens_per_second"] = tps
            
            # Add averaging windows like in online_rl_policy.py
            if len(self.reward_history) >= 10:
                log_data["avg_reward_10"] = np.mean(self.reward_history[-10:])
            if len(self.reward_history) >= 50:
                log_data["avg_reward_50"] = np.mean(self.reward_history[-50:])
            
            wandb.log(log_data)
    
    def _ppo_update(self):
        """Perform PPO update using collected experiences"""
        if len(self.states) < self.batch_size:
            return
        
        # Convert to tensors
        states = torch.stack(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)
        dones = torch.BoolTensor(self.dones).to(self.device)
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for epoch in range(self.ppo_epochs):
            # Create random indices for batch sampling
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    batch_states, batch_actions
                )
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                            self.value_coeff * value_loss + 
                            self.entropy_coeff * entropy_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # Average losses
        n_updates = self.ppo_epochs * (len(states) // self.batch_size)
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy_loss = total_entropy_loss / n_updates
        
        # Store loss history
        self.loss_history.append({
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss
        })
        
        # Update counter
        self.update_count += 1
        
        print(f"PPO Update #{self.update_count}: Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}, Entropy={avg_entropy_loss:.4f}")
        
        # Wandb logging
        if self.use_wandb:
            wandb.log({
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "entropy_loss": avg_entropy_loss,
                "update_count": self.update_count
            })
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation with improved handling"""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # Convert boolean dones to float for arithmetic operations
        dones_float = dones.float()
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state (end of batch)
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones_float[t]) - values[t]
            
            # GAE computation - reset GAE at episode boundaries
            if dones_float[t] == 1.0:
                last_gae = delta  # Reset at episode end
            else:
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            
            advantages[t] = last_gae
        
        return advantages
    
    def save_checkpoint(self, checkpoint_name=None):
        """Save training checkpoint"""
        if checkpoint_name is None:
            checkpoint_name = f"discrete_ppo_checkpoint_step_{self.step_count}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        checkpoint_data = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'update_count': self.update_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'reward_history': self.reward_history,
            'parameter_history': self.parameter_history,
            'loss_history': self.loss_history,
            'tokens_per_second_history': self.tokens_per_second_history,
            'total_tokens_bins': self.total_tokens_bins,
            'depth_bins': self.depth_bins,
            'top_k_bins': self.top_k_bins,
            'valid_actions': self.valid_actions,
            'hyperparameters': {
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'ppo_epochs': self.ppo_epochs,
                'batch_size': self.batch_size,
                'clip_range': self.clip_range,
                'value_coeff': self.value_coeff,
                'entropy_coeff': self.entropy_coeff,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda
            }
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Discrete PPO checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load training checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print(f"❌ No checkpoint found to load")
            return False
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model and optimizer states
            self.network.load_state_dict(checkpoint_data['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Load training state
            self.step_count = checkpoint_data['step_count']
            self.update_count = checkpoint_data['update_count']
            self.questions_processed = checkpoint_data['questions_processed']
            self.training_seed = checkpoint_data.get('training_seed')
            
            # Load performance history
            self.reward_history = checkpoint_data['reward_history']
            self.parameter_history = checkpoint_data['parameter_history']
            self.loss_history = checkpoint_data['loss_history']
            self.tokens_per_second_history = checkpoint_data['tokens_per_second_history']
            
            print(f"✅ Discrete PPO checkpoint loaded: {checkpoint_path}")
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
                          if f.startswith('discrete_ppo_checkpoint_step_') and f.endswith('.pth')]
        if not checkpoint_files:
            return None
        
        # Sort by step number to get latest
        def extract_step(filename):
            try:
                return int(filename.split('_')[-1].split('.')[0])
            except:
                return 0
        
        latest_file = max(checkpoint_files, key=extract_step)
        return os.path.join(self.checkpoint_dir, latest_file)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to keep only max_checkpoints files"""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith('discrete_ppo_checkpoint_step_') and f.endswith('.pth')]
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by step number
        def extract_step(filename):
            try:
                return int(filename.split('_')[-1].split('.')[0])
            except:
                return 0
        
        checkpoint_files.sort(key=extract_step)
        
        # Remove oldest checkpoints
        files_to_remove = checkpoint_files[:-self.max_checkpoints]
        for file_to_remove in files_to_remove:
            file_path = os.path.join(self.checkpoint_dir, file_to_remove)
            os.remove(file_path)
            print(f"🗑️ Removed old checkpoint: {file_to_remove}")
    
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
            'avg_reward': np.mean(self.reward_history),
            'recent_avg_reward': np.mean(recent_rewards),
            'best_reward': max(self.reward_history),
            'questions_processed': self.questions_processed
        }
        
        if self.tokens_per_second_history:
            recent_tps = self.tokens_per_second_history[-100:]
            stats.update({
                'avg_tokens_per_second': np.mean(self.tokens_per_second_history),
                'recent_avg_tokens_per_second': np.mean(recent_tps),
                'best_tokens_per_second': max(self.tokens_per_second_history)
            })
        
        return stats
    
    def save(self, path):
        """Save trained policy (final model)"""
        save_data = {
            'network_state_dict': self.network.state_dict(),
            'total_tokens_bins': self.total_tokens_bins,
            'depth_bins': self.depth_bins,
            'top_k_bins': self.top_k_bins,
            'valid_actions': self.valid_actions,
            'performance_stats': self.get_performance_stats()
        }
        torch.save(save_data, path)
        print(f"💾 Discrete PPO policy saved to: {path}")
    
    def load(self, path):
        """Load trained policy"""
        if not os.path.exists(path):
            print(f"❌ Policy file not found: {path}")
            return False
        
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
            self.network.load_state_dict(data['network_state_dict'])
            
            # Verify parameter bins match
            if (data['total_tokens_bins'] != self.total_tokens_bins or
                data['depth_bins'] != self.depth_bins or 
                data['top_k_bins'] != self.top_k_bins):
                print("⚠️ Warning: Parameter bins in saved model differ from current configuration")
            
            print(f"✅ Discrete PPO policy loaded from: {path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading policy: {e}")
            return False


def calculate_discrete_ppo_reward(generation_time, new_tokens, total_tokens, depth, top_k):
    """Calculate reward for discrete PPO learning with appropriate scale"""
    # Primary reward: tokens per second (speed)
    if generation_time <= 0 or new_tokens <= 0:
        return 0.0
    
    # Reward is simply tokens per second
    tokens_per_second = new_tokens / generation_time
    return tokens_per_second
