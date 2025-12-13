"""
Continuous Online RL Policy for Real-time EAGLE Parameter Optimization
Uses continuous action space for more flexible parameter search
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
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

class ContinuousOnlineTreePolicy:
    """Continuous Online RL Policy using Actor-Critic for parameter optimization"""
    
    def __init__(self, 
                 learning_rate=3e-4,
                 epsilon_start=0.3,  # Lower for continuous exploration
                 epsilon_end=0.02,
                 epsilon_decay=0.995,
                 memory_size=1000,
                 batch_size=32,
                 target_update_freq=100,
                 use_wandb=True,
                 wandb_project="eagle-continuous-rl",
                 wandb_run_name=None,
                 checkpoint_dir="checkpoints",
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
                        "epsilon_start": epsilon_start,
                        "epsilon_end": epsilon_end,
                        "epsilon_decay": epsilon_decay,
                        "memory_size": memory_size,
                        "batch_size": batch_size,
                        "target_update_freq": target_update_freq,
                        "device": str(self.device),
                        "action_space": "continuous"
                    }
                )
                print(f"🔗 Wandb logging initialized: {wandb.run.get_url()}")
            else:
                print(f"🔗 Using existing wandb run: {wandb.run.get_url()}")
        else:
            print("📊 Wandb logging disabled")
        # Continuous parameter ranges
        self.param_ranges = {
            'total_tokens': {'min': 16, 'max': 128},  # Wider range 
            'depth': {'min': 2, 'max': 8},           # Continuous depth
            'top_k': {'min': 2, 'max': 32}           # Continuous top_k
        }
        # self.param_ranges = {
        #     'total_tokens': {'min': 16, 'max': 512},  # Wider range 
        #     'depth': {'min': 2, 'max': 10},           # Continuous depth
        #     'top_k': {'min': 2, 'max': 48}           # Continuous top_k
        # }
        # self.param_ranges = {
        #     'total_tokens': {'min': 16, 'max': 192},  # Wider range 
        #     'depth': {'min': 2, 'max': 10},           # Continuous depth
        #     'top_k': {'min': 2, 'max': 48}           # Continuous top_k
        # }
        # self.param_ranges = {
        #     'total_tokens': {'min': 16, 'max': 256},  # Wider range 
        #     'depth': {'min': 2, 'max': 10},           # Continuous depth
        #     'top_k': {'min': 2, 'max': 48}           # Continuous top_k
        # }
        # self.param_ranges = {
        #     'total_tokens': {'min': 16, 'max': 512},  # Wider range
        #     'depth': {'min': 2, 'max': 10},           # Continuous depth
        #     'top_k': {'min': 2, 'max': 64}           # Continuous top_k
        # }
        
        # Initialize SBERT for state encoding
        print("Loading SBERT model for state representation...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.state_dim = 384
        
        # Initialize Actor-Critic networks
        self.actor_network = self._build_actor_network().to(self.device)
        self.critic_network = self._build_critic_network().to(self.device)
        self.target_critic = self._build_critic_network().to(self.device)
        self.target_critic.load_state_dict(self.critic_network.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=learning_rate)
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Training counters
        self.step_count = 0
        self.target_update_freq = target_update_freq
        
        # Performance tracking
        self.reward_history = []
        self.parameter_history = []
        self.loss_history = []
        self.tokens_per_second_history = []
        
        # Wandb logging setup
        if self.use_wandb:
            wandb.config.update({
                "param_ranges": self.param_ranges,
                "state_dim": self.state_dim,
                "action_space_type": "continuous"
            })
        
        print(f"Continuous Online RL Policy initialized:")
        print(f"  - State dim: {self.state_dim}")
        print(f"  - Action space: Continuous")
        print(f"  - Total tokens: {self.param_ranges['total_tokens']['min']}-{self.param_ranges['total_tokens']['max']}")
        print(f"  - Depth: {self.param_ranges['depth']['min']}-{self.param_ranges['depth']['max']}")
        print(f"  - Top-k: {self.param_ranges['top_k']['min']}-{self.param_ranges['top_k']['max']}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Device: {self.device}")
        print(f"  - Wandb logging: {'enabled' if self.use_wandb else 'disabled'}")
    
    def _build_actor_network(self):
        """Build Actor network that outputs continuous actions (mean and std)"""
        network = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 3 means + 3 log_stds for total_tokens, depth, top_k
        )
        
        # Initialize weights
        for layer in network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
        
        return network
    
    def _build_critic_network(self):
        """Build Critic network that estimates value"""
        network = nn.Sequential(
            nn.Linear(self.state_dim + 3, 256),  # state + actions
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value output
        )
        
        # Initialize weights
        for layer in network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
        
        return network
    
    def _encode_state(self, context):
        """Encode conversation context using SBERT"""
        embedding = self.sbert_model.encode(context)
        return torch.FloatTensor(embedding).to(self.device)
    
    def _normalize_actions(self, raw_actions):
        """Normalize raw network outputs to parameter ranges"""
        # raw_actions: [total_tokens_raw, depth_raw, top_k_raw]
        total_tokens = self._scale_to_range(raw_actions[0], 
                                          self.param_ranges['total_tokens']['min'], 
                                          self.param_ranges['total_tokens']['max'])
        depth = self._scale_to_range(raw_actions[1], 
                                   self.param_ranges['depth']['min'], 
                                   self.param_ranges['depth']['max'])
        top_k = self._scale_to_range(raw_actions[2], 
                                   self.param_ranges['top_k']['min'], 
                                   self.param_ranges['top_k']['max'])
        
        return total_tokens, depth, top_k
    
    def _scale_to_range(self, value, min_val, max_val):
        """Scale a value from [-1, 1] to [min_val, max_val]"""
        # Clamp to prevent extreme values
        value = torch.clamp(value, -3.0, 3.0)
        # Apply tanh to get [-1, 1] range
        normalized = torch.tanh(value)
        # Scale to target range
        scaled = min_val + (normalized + 1) * (max_val - min_val) / 2
        return scaled
    
    def _apply_constraint_and_round(self, total_tokens, depth, top_k):
        """Apply constraint and round to integers"""
        # Round to integers
        total_tokens_int = int(torch.round(total_tokens).item())
        depth_int = int(torch.round(depth).item())
        top_k_int = int(torch.round(top_k).item())
        
        # Apply constraint: total_tokens <= top_k^(depth-1)
        max_possible_tokens = top_k_int ** (depth_int - 1)
        if total_tokens_int > max_possible_tokens:
            total_tokens_int = max_possible_tokens
        
        # Ensure minimum values
        total_tokens_int = max(total_tokens_int, self.param_ranges['total_tokens']['min'])
        depth_int = max(depth_int, self.param_ranges['depth']['min'])
        top_k_int = max(top_k_int, self.param_ranges['top_k']['min'])
        
        return total_tokens_int, depth_int, top_k_int
    
    def predict_parameters(self, context, training_mode=True):
        """Predict parameters using Actor network with exploration"""
        state = self._encode_state(context)
        
        with torch.no_grad():
            # Get action distribution from actor
            actor_output = self.actor_network(state.unsqueeze(0))
            means = actor_output[0, :3]  # First 3 outputs are means
            log_stds = actor_output[0, 3:]  # Last 3 outputs are log standard deviations
            
            # Clamp log_stds to prevent extreme values
            log_stds = torch.clamp(log_stds, -2, 1)
            stds = torch.exp(log_stds)
            
            if training_mode and random.random() < self.epsilon:
                # Exploration: add more noise
                exploration_noise = torch.randn_like(means) * (stds + 0.5)
                raw_actions = means + exploration_noise
                exploration_used = True
            else:
                # Exploitation: sample from learned distribution
                noise = torch.randn_like(means) * stds
                raw_actions = means + noise
                exploration_used = False
        
        # Normalize and apply constraints
        total_tokens_cont, depth_cont, top_k_cont = self._normalize_actions(raw_actions)
        total_tokens, depth, top_k = self._apply_constraint_and_round(
            total_tokens_cont, depth_cont, top_k_cont)
        
        # Store for learning
        if training_mode:
            self.last_state = state
            self.last_raw_actions = raw_actions.detach()
            self.last_actions_continuous = torch.tensor([total_tokens_cont, depth_cont, top_k_cont])
            self.last_params = (total_tokens, depth, top_k)
            self.last_exploration = exploration_used
            
            mode_str = "EXPLORE" if exploration_used else "EXPLOIT"
            # max_tokens = top_k ** (depth - 1)
            print(f"Continuous RL {mode_str} (ε={self.epsilon:.3f}): tt={total_tokens}, d={depth}, k={top_k})")
        
        return total_tokens, depth, top_k
    
    def update_policy(self, reward, generation_time=None, new_tokens=None):
        """Update policy using Actor-Critic learning"""
        if not hasattr(self, 'last_state'):
            return
        
        # Store experience
        experience = {
            'state': self.last_state.detach().cpu(),
            'raw_actions': self.last_raw_actions.detach().cpu(),
            'continuous_actions': self.last_actions_continuous.detach().cpu(),
            'reward': reward,
            'done': True
        }
        self.memory.append(experience)
        
        # Track performance
        self.reward_history.append(reward)
        self.parameter_history.append(self.last_params)
        
        if generation_time and new_tokens:
            tokens_per_sec = new_tokens / generation_time if generation_time > 0 else 0
            self.tokens_per_second_history.append(tokens_per_sec)
        
        # Learn from experience
        if len(self.memory) >= self.batch_size:
            actor_loss, critic_loss = self._learn_from_batch()
            self.loss_history.append({'actor': actor_loss, 'critic': critic_loss})
            print(f"  → Policy updated! Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
        
        # Update exploration
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_critic.load_state_dict(self.critic_network.state_dict())
            print(f"Step {self.step_count}: Updated target network, epsilon={self.epsilon:.3f}")
        
        # Wandb logging
        if self.use_wandb:
            log_dict = {
                "step": self.step_count,
                "reward": reward,
                "epsilon": self.epsilon,
                "memory_size": len(self.memory),
                "total_tokens": self.last_params[0],
                "depth": self.last_params[1],
                "top_k": self.last_params[2],
                "exploration": 1 if self.last_exploration else 0
            }
            
            if generation_time and new_tokens:
                log_dict["tokens_per_second"] = tokens_per_sec
                log_dict["generation_time"] = generation_time
                log_dict["new_tokens"] = new_tokens
            
            if hasattr(self, 'loss_history') and self.loss_history:
                log_dict["actor_loss"] = self.loss_history[-1]['actor']
                log_dict["critic_loss"] = self.loss_history[-1]['critic']
            
            if len(self.reward_history) >= 10:
                log_dict["avg_reward_10"] = np.mean(self.reward_history[-10:])
            
            wandb.log(log_dict)
        
        # Statistics
        if self.step_count % 10 == 0:
            recent_rewards = self.reward_history[-10:]
            avg_reward = np.mean(recent_rewards)
            print(f"Step {self.step_count}: Recent avg reward: {avg_reward:.3f}, ε={self.epsilon:.3f}")
            
            # Show parameter diversity
            recent_params = self.parameter_history[-10:]
            unique_params = len(set(recent_params))
            print(f"  → Parameter diversity: {unique_params}/10 unique combinations")
    
    def _learn_from_batch(self):
        """Actor-Critic learning from experience batch"""
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert to tensors
        states = torch.stack([exp['state'] for exp in batch]).to(self.device)
        raw_actions = torch.stack([exp['raw_actions'] for exp in batch]).to(self.device)
        continuous_actions = torch.stack([exp['continuous_actions'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        
        # Critic loss (TD error)
        state_action = torch.cat([states, continuous_actions], dim=1)
        current_values = self.critic_network(state_action).squeeze()
        
        with torch.no_grad():
            target_values = rewards  # Simple episodic setting
        
        critic_loss = nn.MSELoss()(current_values, target_values)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor loss (policy gradient)
        actor_outputs = self.actor_network(states)
        means = actor_outputs[:, :3]
        log_stds = torch.clamp(actor_outputs[:, 3:], -2, 1)
        stds = torch.exp(log_stds)
        
        # Calculate log probabilities
        dist = torch.distributions.Normal(means, stds)
        log_probs = dist.log_prob(raw_actions).sum(dim=1)
        
        # Advantage estimation
        with torch.no_grad():
            advantages = rewards - current_values.detach()
        
        # Actor loss (negative because we want to maximize)
        actor_loss = -(log_probs * advantages).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    # Add all the checkpoint and utility methods from the discrete version
    def save_checkpoint(self, checkpoint_name=None):
        """Save training checkpoint"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{self.step_count}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        checkpoint_data = {
            'actor_network_state_dict': self.actor_network.state_dict(),
            'critic_network_state_dict': self.critic_network.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'reward_history': self.reward_history,
            'parameter_history': self.parameter_history,
            'loss_history': self.loss_history,
            'tokens_per_second_history': self.tokens_per_second_history,
            'memory': list(self.memory),
            'param_ranges': self.param_ranges,
            'policy_type': 'continuous'
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Continuous checkpoint saved: {checkpoint_path}")
        
        # Log to wandb first before cleanup
        if self.use_wandb and wandb.run is not None:
            try:
                artifact = wandb.Artifact(
                    name=f"continuous_checkpoint_{self.step_count}",
                    type="checkpoint",
                    description=f"Continuous training checkpoint at step {self.step_count}"
                )
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
                print(f"🔗 Continuous checkpoint logged to wandb")
            except Exception as e:
                print(f"⚠️  Failed to log checkpoint: {e}")
        
        # Clean up old checkpoints after wandb logging
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-100:]
        recent_params = self.parameter_history[-100:]
        
        param_stats = {}
        for total_tokens, depth, top_k in recent_params:
            key = f"{total_tokens}-{depth}-{top_k}"
            param_stats[key] = param_stats.get(key, 0) + 1
        
        return {
            'total_episodes': len(self.reward_history),
            'avg_reward_recent': np.mean(recent_rewards),
            'avg_reward_overall': np.mean(self.reward_history),
            'epsilon': self.epsilon,
            'most_used_params': sorted(param_stats.items(), key=lambda x: x[1], reverse=True)[:5],
            'reward_trend': recent_rewards[-10:] if len(recent_rewards) >= 10 else recent_rewards,
            'parameter_diversity': len(set(recent_params)),
            'policy_type': 'continuous'
        }
    
    # Add other utility methods...
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping the most recent ones"""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith('checkpoint_step_') and f.endswith('.pth')]
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        def extract_step(filename):
            try:
                return int(filename.split('_')[2].split('.')[0])
            except:
                return 0
        
        # Sort by step number (oldest first)
        checkpoint_files.sort(key=extract_step)
        
        # Keep only the most recent max_checkpoints files
        files_to_remove = checkpoint_files[:-self.max_checkpoints]
        
        for filename in files_to_remove:
            old_path = os.path.join(self.checkpoint_dir, filename)
            try:
                # Double-check the file exists before removing
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"🗑️  Removed old checkpoint: {filename}")
                else:
                    print(f"⚠️  Checkpoint already removed: {filename}")
            except OSError as e:
                print(f"⚠️  Failed to remove checkpoint {filename}: {e}")
    
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
            'epsilon': self.epsilon,
            'total_episodes': len(self.reward_history),
            'policy_type': 'continuous'
        }
    
    def increment_questions_processed(self, count=1):
        """Track processed questions"""
        self.questions_processed += count
        if self.should_save_checkpoint():
            self.save_checkpoint()
    
    def save(self, save_path):
        """Save the trained policy (interface compatibility)"""
        if save_path:
            # Save the complete policy state
            policy_data = {
                'actor_network_state_dict': self.actor_network.state_dict(),
                'critic_network_state_dict': self.critic_network.state_dict(),
                'target_critic_state_dict': self.target_critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count,
                'questions_processed': self.questions_processed,
                'training_seed': self.training_seed,
                'reward_history': self.reward_history,
                'parameter_history': self.parameter_history,
                'loss_history': self.loss_history,
                'tokens_per_second_history': self.tokens_per_second_history,
                'param_ranges': self.param_ranges,
                'policy_type': 'continuous'
            }
            
            torch.save(policy_data, save_path)
            print(f"💾 Continuous policy saved to: {save_path}")
            
            # Also log final stats
            stats = self.get_performance_stats()
            print(f"📊 Final Performance Stats:")
            print(f"  - Total episodes: {stats.get('total_episodes', 0)}")
            print(f"  - Overall avg reward: {stats.get('avg_reward_overall', 0):.3f}")
            print(f"  - Recent avg reward: {stats.get('avg_reward_recent', 0):.3f}")
            print(f"  - Parameter diversity: {stats.get('parameter_diversity', 0)}")
            print(f"  - Final epsilon: {stats.get('epsilon', 0):.3f}")
        else:
            print("⚠️  No save path provided, skipping policy save")

    def load(self, load_path):
        """Load a trained policy (interface compatibility)"""
        if not os.path.exists(load_path):
            print(f"❌ Policy file not found: {load_path}")
            return False
        
        try:
            checkpoint_data = torch.load(load_path, map_location=self.device)
            
            # Verify this is a continuous policy
            if checkpoint_data.get('policy_type') != 'continuous':
                print(f"❌ Policy type mismatch. Expected 'continuous', got '{checkpoint_data.get('policy_type')}'")
                return False
            
            # Load network states
            self.actor_network.load_state_dict(checkpoint_data['actor_network_state_dict'])
            self.critic_network.load_state_dict(checkpoint_data['critic_network_state_dict'])
            self.target_critic.load_state_dict(checkpoint_data['target_critic_state_dict'])
            
            # Load optimizer states
            self.actor_optimizer.load_state_dict(checkpoint_data['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint_data['critic_optimizer_state_dict'])
            
            # Load training state
            self.epsilon = checkpoint_data['epsilon']
            self.step_count = checkpoint_data['step_count']
            self.questions_processed = checkpoint_data.get('questions_processed', 0)
            self.training_seed = checkpoint_data.get('training_seed')
            
            # Load history
            self.reward_history = checkpoint_data.get('reward_history', [])
            self.parameter_history = checkpoint_data.get('parameter_history', [])
            self.loss_history = checkpoint_data.get('loss_history', [])
            self.tokens_per_second_history = checkpoint_data.get('tokens_per_second_history', [])
            
            # Load parameter ranges (in case they changed)
            if 'param_ranges' in checkpoint_data:
                self.param_ranges = checkpoint_data['param_ranges']
            
            print(f"✅ Continuous policy loaded from: {load_path}")
            print(f"  - Step count: {self.step_count}")
            print(f"  - Episodes: {len(self.reward_history)}")
            print(f"  - Epsilon: {self.epsilon:.3f}")
            print(f"  - Questions processed: {self.questions_processed}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load policy from {load_path}: {e}")
            return False

def calculate_continuous_online_reward(generation_time, new_tokens, total_tokens, depth, top_k):
    """
    Calculate reward for continuous online learning
    Simple reward: directly use speed (tokens/second) as reward
    """
    if generation_time <= 0 or new_tokens <= 0:
        return 0.0
    
    # Reward is simply tokens per second
    tokens_per_second = new_tokens / generation_time
    return tokens_per_second
