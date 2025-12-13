"""
Max-Entropy Online RL Policy for Real-time EAGLE Parameter Optimization
Learns and adapts tree parameters dynamically with enhanced exploration and diversity
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
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

class OnlineTreePolicy:
    """Max-Entropy Online RL Policy that learns tree parameters with enhanced diversity and exploration"""
    
    def __init__(self, 
                 learning_rate=3e-4,
                 epsilon_start=0.9,
                 epsilon_end=0.05,
                 epsilon_decay=0.995,
                 memory_size=1000,
                 batch_size=32,
                 target_update_freq=100,
                 # Max-entropy RL parameters
                 temperature=1.0,          # Temperature for action sampling
                 entropy_weight=0.1,       # Weight for entropy regularization
                 inference_temperature=1.5, # Temperature during inference for diversity
                 max_entropy_inference=True, # Enable max-entropy during inference
                 use_wandb=True,
                 wandb_project="eagle-online-rl",
                 wandb_run_name=None,
                 checkpoint_dir="checkpoints",
                 checkpoint_freq=100,
                 max_checkpoints=3):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Max-entropy RL configuration
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.inference_temperature = inference_temperature
        self.max_entropy_inference = max_entropy_inference
        
        # Resume mechanism configuration
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        self.questions_processed = 0  # Track processed questions for resume
        self.training_seed = None  # Store training seed for reproducible shuffling
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize wandb logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            if not wandb.run:  # Only init if not already initialized
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
                        "temperature": temperature,
                        "entropy_weight": entropy_weight,
                        "inference_temperature": inference_temperature,
                        "max_entropy_inference": max_entropy_inference,
                        "device": str(self.device)
                    }
                )
                print(f"🔗 Wandb logging initialized: {wandb.run.get_url()}")
            else:
                print(f"🔗 Using existing wandb run: {wandb.run.get_url()}")
        else:
            print("📊 Wandb logging disabled")
        
        # Parameter bins for discrete actions - expanded ranges
        self.total_tokens_bins = [32, 48, 64, 80, 96]  # 5 options
        self.depth_bins = [3, 4, 5, 6, 7]  # 5 options
        self.top_k_bins = [4, 8, 12, 16, 20]  # 5 options
        # Expanded parameter bins for more granularity
        self.total_tokens_bins = [16, 32, 48, 64, 80, 96, 128]  # 7 options
        self.depth_bins = [2, 3, 4, 5, 6, 7, 8]  # 7 options
        self.top_k_bins = [2, 4, 8, 12, 16, 20, 32]  # 7 options
        ### Best combination ranges
        self.top_k_bins = [8, 12, 16, 20, 32]
        self.depth_bins = [3, 4, 5, 6, 7, 8]
        self.total_tokens_bins = [32, 48, 64, 80, 96, 128]
        # Action space dimensions
        self.n_total_tokens = len(self.total_tokens_bins)
        self.n_depth = len(self.depth_bins)
        self.n_top_k = len(self.top_k_bins)
        self.total_actions = self.n_total_tokens * self.n_depth * self.n_top_k
        
        # Precompute valid actions to avoid runtime constraint violations
        # Constraint: total_tokens <= top_k^(depth-1)
        self.valid_actions = self._precompute_valid_actions()
        print(f"Action space: {self.total_actions} total, {len(self.valid_actions)} valid")
        
        # Initialize SBERT for state encoding
        print("Loading SBERT model for state representation...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.state_dim = 384  # SBERT embedding dimension
        
        # Initialize Q-network and target network
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
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
        
        # Performance tracking with wandb integration (including entropy metrics)
        self.reward_history = []
        self.parameter_history = []
        self.loss_history = []
        self.tokens_per_second_history = []
        self.entropy_history = []  # Track entropy for max-entropy RL
        self.action_diversity_history = []  # Track action diversity
        
        # Wandb logging setup
        if self.use_wandb:
            # Log initial configuration
            wandb.config.update({
                "total_actions": self.total_actions,
                "valid_actions": len(self.valid_actions),
                "valid_coverage": len(self.valid_actions)/self.total_actions,
                "total_tokens_bins": self.total_tokens_bins,
                "depth_bins": self.depth_bins,
                "top_k_bins": self.top_k_bins,
                "temperature": self.temperature,
                "entropy_weight": self.entropy_weight,
                "inference_temperature": self.inference_temperature,
                "max_entropy_inference": self.max_entropy_inference
            })
        
        print(f"Max-Entropy Online RL Policy initialized:")
        print(f"  - State dim: {self.state_dim}")
        print(f"  - Action space: {self.total_actions} total ({self.n_total_tokens}x{self.n_depth}x{self.n_top_k})")
        print(f"  - Valid actions: {len(self.valid_actions)} ({len(self.valid_actions)/self.total_actions*100:.1f}%)")
        print(f"  - Constraint: total_tokens ≤ top_k^(depth-1)")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Max-entropy config:")
        print(f"    • Training temperature: {self.temperature}")
        print(f"    • Entropy weight: {self.entropy_weight}")
        print(f"    • Inference temperature: {self.inference_temperature}")
        print(f"    • Max-entropy inference: {self.max_entropy_inference}")
        print(f"  - Device: {self.device}")
        print(f"  - Wandb logging: {'enabled' if self.use_wandb else 'disabled'}")
        print(f"  - Valid actions: {len(self.valid_actions)} ({len(self.valid_actions)/self.total_actions*100:.1f}%)")
        print(f"  - Constraint: total_tokens ≤ top_k^(depth-1)")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Device: {self.device}")
        print(f"  - Wandb logging: {'enabled' if self.use_wandb else 'disabled'}")
        print(f"valid: tt≤k^(d-1)")
    
    def _build_network(self):
        """Build Q-network architecture with better initialization"""
        network = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.total_actions)
        )
        
        # Better weight initialization to encourage exploration
        for layer in network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)  # Small positive bias
        
        return network
    @torch.no_grad()
    def _encode_state(self, context):
        """Encode conversation context using SBERT"""
        with torch.no_grad():
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
    
    def predict_parameters(self, context, training_mode=True):
        """Predict parameters using max-entropy policy with temperature-based sampling and enhanced exploration"""
        state = self._encode_state(context)
        
        # Max-entropy action selection with temperature-based sampling
        if self.max_entropy_inference and not training_mode:
            # Inference mode with max-entropy: use temperature-based sampling for diversity
            action, entropy = self._sample_action_with_temperature(state, self.inference_temperature)
            exploration_mode = "MAX-ENTROPY"
            
            # Log entropy for monitoring
            self.entropy_history.append(entropy)
            if self.use_wandb:
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
        
        # Safety backup: clamp parameters if somehow invalid (should never happen with valid actions)
        total_tokens, depth, top_k = self._safe_clamp_params(total_tokens, depth, top_k)
        
        # Store state-action for learning
        if training_mode:
            self.last_state = state
            self.last_action = action
            self.last_params = (total_tokens, depth, top_k)
            self.last_exploration_mode = exploration_mode
            self.last_entropy = entropy
            
            # Debug print for training mode with entropy info
            # max_tokens = top_k ** (depth - 1)
            # print(f"Max-Entropy RL {exploration_mode} (ε={self.epsilon:.3f}, H={entropy:.3f}): tt={total_tokens}, d={depth}, k={top_k} (max={max_tokens})")
        else:
            pass
            # Inference mode
            # max_tokens = top_k ** (depth - 1)
            # print(f"Max-Entropy RL {exploration_mode} (T={self.inference_temperature:.1f}, H={entropy:.3f}): tt={total_tokens}, d={depth}, k={top_k} (max={max_tokens})")
        
        return total_tokens, depth, top_k
    
    def _sample_action_with_temperature(self, state, temperature):
        """Sample action using temperature-based softmax for max-entropy exploration"""
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            
            # Mask invalid actions by setting their Q-values to -inf
            masked_q_values = q_values.clone()
            for i in range(self.total_actions):
                if i not in self.valid_actions:
                    masked_q_values[0, i] = float('-inf')
            
            if temperature > 0:
                # Apply temperature scaling
                scaled_q_values = masked_q_values / temperature
                # Use softmax to get action probabilities
                action_probs = torch.softmax(scaled_q_values, dim=1)
                
                # Sample from the distribution
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
                
                # Calculate entropy for monitoring
                entropy = action_dist.entropy().item()
                
                return action, entropy
            else:
                # Deterministic (temperature = 0)
                action = masked_q_values.argmax().item()
                return action, 0.0
    
    def update_policy(self, reward, generation_time=None, new_tokens=None):
        """Update policy with reward from last action including entropy regularization"""
        if not hasattr(self, 'last_state'):
            return
        
        # Apply entropy regularization to reward for max-entropy RL
        if hasattr(self, 'last_entropy') and self.entropy_weight > 0:
            entropy_bonus = self.entropy_weight * self.last_entropy
            augmented_reward = reward + entropy_bonus
            # print(f"  → Entropy bonus: {entropy_bonus:.3f} (H={self.last_entropy:.3f}, w={self.entropy_weight})")
        else:
            augmented_reward = reward
        
        # Store experience in replay buffer
        experience = {
            'state': self.last_state.detach().cpu(),  # Detach to avoid gradient issues
            'action': self.last_action,
            'reward': augmented_reward,  # Use entropy-augmented reward
            'original_reward': reward,   # Store original for analysis
            'done': True  # Each inference is treated as terminal
        }
        self.memory.append(experience)
        
        # Track performance
        self.reward_history.append(reward)  # Track original reward
        self.parameter_history.append(self.last_params)
        
        # Track action diversity
        recent_params = self.parameter_history[-20:] if len(self.parameter_history) >= 20 else self.parameter_history
        unique_params = len(set(recent_params))
        self.action_diversity_history.append(unique_params / len(recent_params))
        
        # Track tokens per second if provided
        if generation_time and new_tokens:
            tokens_per_sec = new_tokens / generation_time if generation_time > 0 else 0
            self.tokens_per_second_history.append(tokens_per_sec)
        
        # Debug info for learning with entropy info
        explore_str = getattr(self, 'last_exploration_mode', 'UNKNOWN')
        # entropy_str = f", H={getattr(self, 'last_entropy', 0):.3f}" if hasattr(self, 'last_entropy') else ""
        # print(f"  → Reward: {reward:.3f} (aug: {augmented_reward:.3f}) for {self.last_params} ({explore_str}{entropy_str})")
        
        # Learn from experience if we have enough samples
        if len(self.memory) >= self.batch_size:
            loss = self._learn_from_batch()
            self.loss_history.append(loss)
            # print(f"  → Policy updated! Memory: {len(self.memory)}/{self.memory.maxlen}")
        
        # Update exploration rate
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # if abs(old_epsilon - self.epsilon) > 0.001:
            # print(f"  → Exploration decreased: {old_epsilon:.3f} → {self.epsilon:.3f}")
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            # print(f"Step {self.step_count}: Updated target network, epsilon={self.epsilon:.3f}")
        
        # Wandb logging for real-time tracking with entropy metrics
        if self.use_wandb:
            log_dict = {
                "step": self.step_count,
                "reward": reward,
                "augmented_reward": augmented_reward,
                "epsilon": self.epsilon,
                "memory_size": len(self.memory),
                "total_tokens": self.last_params[0],
                "depth": self.last_params[1], 
                "top_k": self.last_params[2],
                "exploration_mode": explore_str
            }
            
            # Add entropy metrics
            if hasattr(self, 'last_entropy'):
                log_dict.update({
                    "action_entropy": self.last_entropy,
                    "entropy_bonus": entropy_bonus if hasattr(self, 'last_entropy') and self.entropy_weight > 0 else 0
                })
            
            # Add diversity metrics
            if self.action_diversity_history:
                log_dict["action_diversity"] = self.action_diversity_history[-1]
            
            # Add tokens per second if available
            if generation_time and new_tokens:
                log_dict["tokens_per_second"] = tokens_per_sec
                log_dict["generation_time"] = generation_time
                log_dict["new_tokens"] = new_tokens
            
            # Add loss if available
            if hasattr(self, 'loss_history') and self.loss_history:
                log_dict["loss"] = self.loss_history[-1]
            
            # Add recent performance metrics
            if len(self.reward_history) >= 10:
                log_dict["avg_reward_10"] = torch.mean(torch.tensor(self.reward_history[-10:])).item()
                log_dict["avg_tokens_per_second_10"] = torch.mean(torch.tensor(self.tokens_per_second_history[-10:])).item()
            if len(self.reward_history) >= 50:
                log_dict["avg_reward_50"] = torch.mean(torch.tensor(self.reward_history[-50:])).item()
                log_dict["avg_tokens_per_second_50"] = torch.mean(torch.tensor(self.tokens_per_second_history[-50:])).item()

            wandb.log(log_dict)
            
        # Show statistics every 10 steps
        if self.step_count % 10 == 0:
            recent_rewards = self.reward_history[-10:]
            avg_reward = torch.mean(torch.tensor(recent_rewards)).item()
            # print(f"Step {self.step_count}: Recent avg reward: {avg_reward:.3f}, ε={self.epsilon:.3f}")
            
            # Show parameter diversity
            recent_params = self.parameter_history[-10:]
            unique_params = len(set(recent_params))
            # print(f"  → Parameter diversity: {unique_params}/10 unique combinations")
            
    def save_checkpoint(self, checkpoint_name=None):
        """Save training checkpoint with automatic cleanup of old checkpoints"""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_step_{self.step_count}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save complete training state
        checkpoint_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'reward_history': self.reward_history,
            'parameter_history': self.parameter_history,
            'loss_history': self.loss_history,
            'tokens_per_second_history': self.tokens_per_second_history,
            'entropy_history': self.entropy_history,
            'action_diversity_history': self.action_diversity_history,
            'memory': list(self.memory),  # Convert deque to list for serialization
            'total_tokens_bins': self.total_tokens_bins,
            'depth_bins': self.depth_bins,
            'top_k_bins': self.top_k_bins,
            'valid_actions': self.valid_actions,
            # Training configuration for consistency
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'memory_size': self.memory.maxlen,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'checkpoint_freq': self.checkpoint_freq,
            'max_checkpoints': self.max_checkpoints,
            # Max-entropy RL configuration
            'temperature': self.temperature,
            'entropy_weight': self.entropy_weight,
            'inference_temperature': self.inference_temperature,
            'max_entropy_inference': self.max_entropy_inference
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        # print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Log checkpoint to wandb if enabled
        if self.use_wandb and wandb.run is not None:
            try:
                artifact = wandb.Artifact(
                    name=f"training_checkpoint_{self.step_count}",
                    type="checkpoint",
                    description=f"Training checkpoint at step {self.step_count}"
                )
                artifact.add_file(checkpoint_path)
                artifact.metadata = {
                    "step_count": self.step_count,
                    "questions_processed": self.questions_processed,
                    "epsilon": self.epsilon,
                    "avg_reward_recent": torch.mean(torch.tensor(self.reward_history[-10:])).item() if len(self.reward_history) >= 10 else torch.mean(torch.tensor(self.reward_history)).item() if self.reward_history else 0
                }
                wandb.log_artifact(artifact)
                # print(f"🔗 Checkpoint logged to wandb: {artifact.name}")
            except Exception as e:
                print(f"⚠️  Failed to log checkpoint artifact to wandb: {e}")
                print("   Checkpoint still saved locally successfully")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load training checkpoint to resume from interruption"""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print(f"❌ No checkpoint found at {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Check for action space compatibility before loading model states
            checkpoint_total_actions = checkpoint.get('total_actions', None)
            if checkpoint_total_actions is not None and checkpoint_total_actions != self.total_actions:
                print(f"❌ Action space mismatch: checkpoint has {checkpoint_total_actions} actions, current model has {self.total_actions}")
                print(f"   This usually happens when parameter bins have been changed.")
                print(f"   Skipping checkpoint loading and starting fresh.")
                return False
            
            # Restore model states
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training progress
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.questions_processed = checkpoint.get('questions_processed', 0)
            self.training_seed = checkpoint.get('training_seed', None)
            
            # Restore history
            self.reward_history = checkpoint['reward_history']
            self.parameter_history = checkpoint['parameter_history']
            self.loss_history = checkpoint.get('loss_history', [])
            self.tokens_per_second_history = checkpoint.get('tokens_per_second_history', [])
            self.entropy_history = checkpoint.get('entropy_history', [])
            self.action_diversity_history = checkpoint.get('action_diversity_history', [])
            
            # Restore max-entropy RL parameters
            self.temperature = checkpoint.get('temperature', 1.0)
            self.entropy_weight = checkpoint.get('entropy_weight', 0.1)
            self.inference_temperature = checkpoint.get('inference_temperature', 1.5)
            self.max_entropy_inference = checkpoint.get('max_entropy_inference', True)
            
            # Restore experience replay memory
            if 'memory' in checkpoint:
                self.memory.clear()
                for experience in checkpoint['memory']:
                    self.memory.append(experience)
            
            # Restore valid actions and parameter bins
            if 'valid_actions' in checkpoint:
                self.valid_actions = checkpoint['valid_actions']
            if 'total_tokens_bins' in checkpoint:
                self.total_tokens_bins = checkpoint['total_tokens_bins']
                self.depth_bins = checkpoint['depth_bins']
                self.top_k_bins = checkpoint['top_k_bins']
            
            print(f"✅ Checkpoint loaded: {checkpoint_path}")
            print(f"   Step: {self.step_count}, Questions processed: {self.questions_processed}")
            print(f"   Epsilon: {self.epsilon:.3f}, Memory size: {len(self.memory)}")
            print(f"   Training seed: {self.training_seed}")
            print(f"   Max-entropy RL: temp={self.temperature:.2f}, ent_weight={self.entropy_weight:.3f}")
            print(f"   Inference temp: {self.inference_temperature:.2f}, max-entropy mode: {self.max_entropy_inference}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
            return False
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint file in checkpoint directory"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_step_') and f.endswith('.pth')]
        if not checkpoint_files:
            return None
        
        # Sort by step number to get latest
        def extract_step(filename):
            try:
                return int(filename.split('_')[2].split('.')[0])
            except:
                return 0
        
        latest_file = max(checkpoint_files, key=extract_step)
        return os.path.join(self.checkpoint_dir, latest_file)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to keep only max_checkpoints files"""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_step_') and f.endswith('.pth')]
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by step number
        def extract_step(filename):
            try:
                return int(filename.split('_')[2].split('.')[0])
            except:
                return 0
        
        checkpoint_files.sort(key=extract_step)
        
        # Remove oldest checkpoints
        files_to_remove = checkpoint_files[:-self.max_checkpoints]
        for filename in files_to_remove:
            old_path = os.path.join(self.checkpoint_dir, filename)
            try:
                os.remove(old_path)
                print(f"🗑️  Removed old checkpoint: {filename}")
            except OSError as e:
                print(f"⚠️  Failed to remove old checkpoint {filename}: {e}")
    
    def should_save_checkpoint(self):
        """Check if it's time to save a checkpoint"""
        return self.step_count > 0 and self.step_count % self.checkpoint_freq == 0
    
    def set_training_seed(self, seed):
        """Set the training seed for reproducible shuffling"""
        self.training_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"🎲 Training seed set to: {seed}")
    
    def get_resume_info(self):
        """Get information needed for resuming training"""
        return {
            'step_count': self.step_count,
            'questions_processed': self.questions_processed,
            'training_seed': self.training_seed,
            'epsilon': self.epsilon,
            'total_episodes': len(self.reward_history)
        }
    
    def increment_questions_processed(self, count=1):
        """Track processed questions for resume capability"""
        self.questions_processed += count
        
        # Auto-save checkpoint if needed
        if self.should_save_checkpoint():
            self.save_checkpoint()
    
    def _learn_from_batch(self):
        """Learn from a batch of experiences using DQN with valid action masking"""
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert states to tensor, handling both tensor and numpy array inputs
        state_list = []
        for exp in batch:
            state = exp['state']
            if isinstance(state, torch.Tensor):
                # Already a tensor, just ensure it's on CPU and convert to numpy
                state_list.append(state.cpu().numpy())
            else:
                # Already a numpy array
                state_list.append(state)
        
        state_data = np.array(state_list)
        states = torch.FloatTensor(state_data).to(self.device)  # No requires_grad needed for inputs
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        
        # Current Q-values (Q-network parameters will have gradients automatically)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values with valid action masking (detached, no gradients needed)
        with torch.no_grad():
            target_q_all = self.target_network(states)
            
            # Mask invalid actions in target network
            for i in range(target_q_all.shape[0]):  # For each sample in batch
                for j in range(self.total_actions):
                    if j not in self.valid_actions:
                        target_q_all[i, j] = float('-inf')
            
            # Use rewards directly as targets (episodic setting)
            target_q_values = rewards.unsqueeze(1)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        if self.step_count % 50 == 0:
            avg_reward = torch.mean(torch.tensor(self.reward_history[-50:])).item() if self.reward_history else 0
            print(f"Step {self.step_count}: Loss={loss.item():.4f}, Avg Reward={avg_reward:.4f}, Valid Actions: {len(self.valid_actions)}")
            
            # Show constraint compliance
            # recent_params = self.parameter_history[-20:] if len(self.parameter_history) >= 20 else self.parameter_history
            # violations = sum(1 for tt, d, k in recent_params if tt > k**(d-1))
            # print(f"  → Constraint violations in last {len(recent_params)} actions: {violations} ({violations/len(recent_params)*100 if len(recent_params) != 0 else 0:.1f}%)")
        
        return loss.item()  # Return loss value for logging
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-100:]
        recent_params = self.parameter_history[-100:]
        
        # Parameter usage statistics
        param_stats = {}
        for i, (total_tokens, depth, top_k) in enumerate(recent_params):
            key = f"{total_tokens}-{depth}-{top_k}"
            param_stats[key] = param_stats.get(key, 0) + 1
        
        return {
            'total_episodes': len(self.reward_history),
            'avg_reward_recent': torch.mean(torch.tensor(recent_rewards)).item(),
            'avg_reward_overall': torch.mean(torch.tensor(self.reward_history)).item(),
            'epsilon': self.epsilon,
            'most_used_params': sorted(param_stats.items(), key=lambda x: x[1], reverse=True)[:5],
            'reward_trend': recent_rewards[-10:] if len(recent_rewards) >= 10 else recent_rewards
        }
    
    def save(self, path):
        """Save the trained policy with valid actions and wandb artifacts (final save)"""
        # Use checkpoint save for consistency but mark as final
        final_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
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
            'is_final_model': True  # Mark as final trained model
        }
        
        torch.save(final_data, path)
        print(f"✅ Final trained policy saved to {path} (with {len(self.valid_actions)} valid actions)")
        
        # Save as wandb artifact
        if self.use_wandb and wandb.run is not None:
            try:
                artifact = wandb.Artifact(
                    name="final_online_rl_policy",
                    type="model",
                    description="Final trained EAGLE Online RL Policy with constraint validation"
                )
                artifact.add_file(path)
                
                # Add metadata
                artifact.metadata = {
                    "final_epsilon": self.epsilon,
                    "total_steps": self.step_count,
                    "questions_processed": self.questions_processed,
                    "final_avg_reward": torch.mean(torch.tensor(self.reward_history[-50:])).item() if len(self.reward_history) >= 50 else torch.mean(torch.tensor(self.reward_history)).item(),
                    "parameter_diversity": len(set(self.parameter_history[-100:])) if len(self.parameter_history) >= 100 else len(set(self.parameter_history)),
                    "valid_actions": len(self.valid_actions),
                    "total_actions": self.total_actions,
                    "training_seed": self.training_seed
                }
                
                wandb.log_artifact(artifact)
                print(f"🔗 Final policy saved as wandb artifact: {artifact.name}")
            except Exception as e:
                print(f"⚠️  Failed to log final policy artifact to wandb: {e}")
                print("   Policy still saved locally successfully")
    
    def load(self, path):
        """Load a trained policy with valid actions (supports both checkpoints and final models)"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # Load core model state
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            
            # Load resume information if available
            self.questions_processed = checkpoint.get('questions_processed', 0)
            self.training_seed = checkpoint.get('training_seed', None)
            
            # Load training history
            self.reward_history = checkpoint['reward_history']
            self.parameter_history = checkpoint['parameter_history']
            
            # Load additional history if available
            if 'loss_history' in checkpoint:
                self.loss_history = checkpoint['loss_history']
            if 'tokens_per_second_history' in checkpoint:
                self.tokens_per_second_history = checkpoint['tokens_per_second_history']
            
            # Restore experience replay memory if available (for checkpoints)
            if 'memory' in checkpoint:
                self.memory.clear()
                for experience in checkpoint['memory']:
                    self.memory.append(experience)
            
            # Handle backward compatibility for valid_actions
            if 'valid_actions' in checkpoint:
                self.valid_actions = checkpoint['valid_actions']
                model_type = "final model" if checkpoint.get('is_final_model', False) else "checkpoint"
                print(f"✅ Online policy loaded from {path} ({model_type} with {len(self.valid_actions)} valid actions)")
                if self.training_seed:
                    print(f"   Training seed: {self.training_seed}, Questions processed: {self.questions_processed}")
            else:
                # Recompute if not saved (backward compatibility)
                self.valid_actions = self._precompute_valid_actions()
                print(f"✅ Online policy loaded from {path} (legacy format, recomputed {len(self.valid_actions)} valid actions)")
            
            return True
        return False
    
    def _precompute_valid_actions(self):
        """Precompute valid actions that satisfy: total_tokens <= top_k^(depth-1)"""
        valid_actions = []
        constraint_violations = 0
        
        for action in range(self.total_actions):
            total_tokens, depth, top_k = self._action_to_params(action)
            max_possible_tokens = top_k ** (depth - 1)
            
            if total_tokens <= max_possible_tokens:
                valid_actions.append(action)
            else:
                constraint_violations += 1
        
        print(f"Constraint analysis: {constraint_violations}/{self.total_actions} actions violate tt <= k^(d-1)")
        print(f"Valid action coverage: {len(valid_actions)/self.total_actions*100:.1f}%")
        
        return valid_actions
    
    def _is_valid_combination(self, total_tokens, depth, top_k):
        """Check if parameter combination satisfies constraint: total_tokens <= top_k^(depth-1)"""
        max_possible_tokens = top_k ** (depth - 1)
        return total_tokens <= max_possible_tokens
    
    def _safe_clamp_params(self, total_tokens, depth, top_k):
        """Safely clamp total_tokens to satisfy constraint as backup safety measure"""
        max_possible_tokens = top_k ** (depth - 1)
        if total_tokens > max_possible_tokens:
            clamped_total_tokens = max_possible_tokens
            print(f"⚠️  Safety clamp: {total_tokens} → {clamped_total_tokens} (depth={depth}, top_k={top_k})")
            return clamped_total_tokens, depth, top_k
        return total_tokens, depth, top_k


def calculate_online_reward(generation_time, new_tokens, total_tokens, depth, top_k):
    """
    Calculate reward for online learning
    Simple reward: directly use speed (tokens/second) as reward
    """
    if generation_time <= 0 or new_tokens <= 0:
        return 0.0
    
    # Reward is simply tokens per second
    tokens_per_second = new_tokens / generation_time
    return tokens_per_second
