"""Generate answers with local models.

Usage:
zpython    # Enhanced question handling for online RL training vs inference
    if args.use_online_rl and not args.online_inference_only:
        # Training mode: repeat and shuffle questions for diverse training data
        original_count = len(questions)
        repeat_factor = args.online_repeat_factor
        questions = questions * repeat_factor
        
        # Shuffle the repeated questions for diverse training order
        import random
        random.shuffle(questions)
        
        print(f"🔄 Online RL Training Mode: Expanded {original_count} → {len(questions)} questions (repeat={repeat_factor}, shuffled)")
    else:
        # Inference mode or non-online-rl: keep original order
        print(f"📋 Standard Mode: Using {len(questions)} questions in original order")wer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import wandb
import signal
import subprocess
import threading
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from accelerate.utils import set_seed
set_seed(0)

import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
    from .rl_tree_policy import RLTreePolicy, calculate_real_reward
    from .online_rl_policy import OnlineTreePolicy, calculate_online_reward
    from .continuous_online_rl_policy import ContinuousOnlineTreePolicy, calculate_continuous_online_reward
    from .ppo_online_rl_policy import PPOOnlineTreePolicy, calculate_ppo_online_reward
    from .discrete_ppo_online_rl_policy import DiscretePPOOnlineTreePolicy, calculate_discrete_ppo_reward
    from .sb3_discrete_ppo_online_rl_policy import SB3DiscretePPOOnlineTreePolicy, calculate_sb3_discrete_ppo_reward
    # OPTIMIZED POLICIES
    from .optimized_sb3_discrete_ppo_online_rl_policy import CustomPPOOnlineTreePolicy, calculate_custom_ppo_reward
    from .optimized_sb3_discrete_ppo_online_rl_policy_ofl import OptimizedSB3DiscretePPOOnlineTreePolicy as OptimizedSB3DiscretePPOOnlineTreePolicyOFL
    from .optimized_online_rl_policy import OptimizedOnlineTreePolicy, calculate_optimized_online_reward
except:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import *
    from eagle.evaluation.rl_tree_policy import RLTreePolicy, calculate_real_reward
    from eagle.evaluation.online_rl_policy import OnlineTreePolicy, calculate_online_reward
    from eagle.evaluation.continuous_online_rl_policy import ContinuousOnlineTreePolicy, calculate_continuous_online_reward
    from eagle.evaluation.ppo_online_rl_policy import PPOOnlineTreePolicy, calculate_ppo_online_reward
    from eagle.evaluation.discrete_ppo_online_rl_policy import DiscretePPOOnlineTreePolicy, calculate_discrete_ppo_reward
    from eagle.evaluation.sb3_discrete_ppo_online_rl_policy import SB3DiscretePPOOnlineTreePolicy, calculate_sb3_discrete_ppo_reward
    # OPTIMIZED POLICIES
    from eagle.evaluation.optimized_sb3_discrete_ppo_online_rl_policy import CustomPPOOnlineTreePolicy, calculate_custom_ppo_reward
    from eagle.evaluation.optimized_sb3_discrete_ppo_online_rl_policy_ofl import OptimizedSB3DiscretePPOOnlineTreePolicy as OptimizedSB3DiscretePPOOnlineTreePolicyOFL
    from eagle.evaluation.optimized_online_rl_policy import OptimizedOnlineTreePolicy, calculate_optimized_online_reward


def detect_actual_gpu_usage():
    """
    Detect if multiple GPUs are actually being used by checking:
    1. CUDA_VISIBLE_DEVICES environment variable
    2. Available GPU count via torch.cuda
    3. Ray distributed setup
    """
    import torch
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        visible_gpus = [x.strip() for x in cuda_visible.split(',') if x.strip()]
        if len(visible_gpus) > 1:
            print(f"🔍 Detected multi-GPU via CUDA_VISIBLE_DEVICES: {visible_gpus}")
            return True
    
    # Check torch.cuda device count
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"🔍 Detected {gpu_count} available GPUs via torch.cuda")
            return True
    
    # Check if Ray is being used (indication of distributed setup)
    try:
        import ray
        if ray.is_initialized():
            print(f"🔍 Detected Ray distributed setup (likely multi-GPU)")
            return True
    except ImportError:
        pass
    
    print(f"🔍 Single GPU or CPU setup detected")
    return False


def wandb_login_with_timeout(api_key, timeout=10):
    """
    Attempt wandb login with timeout to avoid hanging on DNS issues.
    """
    
    def login_worker(result_container):
        try:
            wandb.login(key=api_key)
            result_container['success'] = True
        except Exception as e:
            result_container['error'] = str(e)
    
    result = {'success': False, 'error': None}
    thread = threading.Thread(target=login_worker, args=(result,))
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # Thread is still running, login timed out
        print(f"⏱️  wandb login timed out after {timeout}s (likely DNS/network issue)")
        return False
    
    if result['success']:
        return True
    else:
        print(f"❌ wandb login failed: {result.get('error', 'Unknown error')}")
        return False


def parse_ppo_net_arch_args(net_arch_str, policy_version="standard"):
    """
    Parse network architecture string for PPO policies.
    
    Args:
        net_arch_str: String representation of network architecture
        policy_version: "standard" or "ofl"
    
    Returns:
        For standard version: list of integers [512, 256, 128]
        For OFL version: dict {"pi": [128, 128], "vf": [128, 128]} or list [128, 128] for both
    """
    net_arch_str = net_arch_str.strip().strip('"').strip("'")
    if not net_arch_str or net_arch_str.strip() == "":
        if policy_version == "standard":
            return [64, 64]  # Default for standard version
        else:
            return [64, 64]  # Default for OFL version
    
    if policy_version == "standard":
        # Parse comma-separated integers for standard version
        try:
            return [int(x.strip()) for x in net_arch_str.split(",")]
        except ValueError:
            print(f"Warning: Invalid network architecture '{net_arch_str}' for standard version. Using default [64, 64]")
            return [64, 64]
    else:
        # Parse OFL version format: "pi_layers;vf_layers" or just "layers"
        if ";" in net_arch_str:
            # Format: "64,64;128,128"
            try:
                pi_str, vf_str = net_arch_str.split(";", 1)
                pi_layers = [int(x.strip()) for x in pi_str.split(",")]
                vf_layers = [int(x.strip()) for x in vf_str.split(",")]
                return {"pi": pi_layers, "vf": vf_layers}
            except (ValueError, IndexError):
                print(f"Warning: Invalid network architecture '{net_arch_str}' for OFL version. Using default [64, 64]")
                return [64, 64]
        else:
            # Format: "64,64" (same for both pi and vf)
            try:
                layers = [int(x.strip()) for x in net_arch_str.split(",")]
                return layers
            except ValueError:
                print(f"Warning: Invalid network architecture '{net_arch_str}' for OFL version. Using default [64, 64]")
                return [64, 64]


def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    questions = load_questions(question_file, question_begin, question_end)
    
    # Enhanced question handling for online RL training vs inference with resume support
    if args.use_online_rl and not args.online_inference_only:
        # Training mode: repeat and shuffle questions for diverse training data
        original_count = len(questions)
        repeat_factor = args.online_repeat_factor  # Repeat questions for more training data
        questions = questions * repeat_factor
        
        # Use seed-based shuffling for reproducible training order (needed for resume)
        import random
        training_seed = args.training_seed if hasattr(args, 'training_seed') and args.training_seed else 42
        random.seed(training_seed)
        random.shuffle(questions)
        
        print(f"🔄 Online RL Training Mode: Expanded {original_count} → {len(questions)} questions (repeat={repeat_factor}, seed={training_seed})")
    else:
        # Inference mode or non-online-rl: keep original order
        print(f"📋 Standard Mode: Using {len(questions)} questions in original order")
    
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    # temperature = 0.0

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto",
        use_eagle3=args.use_eagle3,
    )

    tokenizer = model.get_tokenizer()

    # Set max_length parameter once to avoid repeated computation
    max_length_param = 2200 if (args.bench_name == "sum" or "sum" in args.question_file) and args.online_inference_only else 2048

    # Initialize RL policy if requested
    rl_policy = None
    online_policy = None
    rl_data_entries = []
    
    if args.use_online_rl:
        # Initialize online RL policy for real-time learning with resume support
        print("Initializing Online RL Policy for real-time learning...")
        
        # Setup checkpoint directory
        checkpoint_dir = args.checkpoint_dir if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir else "checkpoints"
        
        # Detect if running on multiple GPUs to avoid wandb connectivity issues
        multi_gpu_detected = detect_actual_gpu_usage()
        
        # Setup wandb authentication if needed (with timeout protection)
        use_wandb = not args.online_inference_only and not args.no_wandb
        wandb_login_successful = False
        
        if use_wandb:
            # Legacy override: force disable wandb on multi-GPU if explicitly requested
            if multi_gpu_detected and getattr(args, 'force_disable_wandb_multi_gpu', False):
                print("🔧 Multi-GPU setup detected: wandb disabled via --force-disable-wandb-multi-gpu")
                print("   Note: You can now use wandb with timeout on multi-GPU setups by omitting this flag")
                use_wandb = False
            else:
                try:
                    
                    # Check if wandb is already disabled via environment
                    wandb_mode = os.environ.get('WANDB_MODE', '').lower()
                    if wandb_mode in ['disabled', 'offline']:
                        print(f"🔧 wandb is disabled via WANDB_MODE={wandb_mode}")
                        use_wandb = False
                    else:
                        # Use environment variable for API key (set WANDB_API_KEY in your environment)
                        wandb_api_key = os.environ.get('WANDB_API_KEY')
                        if wandb_api_key:
                            # Always attempt wandb login with timeout, regardless of multi-GPU setup
                            timeout_seconds = getattr(args, 'wandb_timeout', 10)
                            gpu_info = f" (Multi-GPU: {multi_gpu_detected})" if multi_gpu_detected else ""
                            print(f"🔗 Attempting wandb authentication{gpu_info} (with {timeout_seconds}s timeout)...")
                            if wandb_login_with_timeout(wandb_api_key, timeout=timeout_seconds):
                                print("✅ wandb authentication successful")
                                wandb_login_successful = True
                            else:
                                print("⚠️  wandb authentication timed out or failed")
                                if multi_gpu_detected:
                                    print("   This is common on multi-GPU setups with network restrictions")
                                print("   Continuing without wandb logging...")
                                use_wandb = False
                        else:
                            print("⚠️  WANDB_API_KEY environment variable not set")
                            print("   Set it with: export WANDB_API_KEY=your_api_key")
                            print("   Or disable with: export WANDB_MODE=disabled")
                            print("   Continuing without wandb logging...")
                            use_wandb = False
                except Exception as e:
                    print(f"⚠️  wandb setup failed: {e}")
                    print("   Continuing without wandb logging...")
                    use_wandb = False
        
        wandb_run_name = f"eagle-online-{args.model_id}-{int(time.time())}" if use_wandb else None
        
        # Choose between PPO, continuous, and discrete action space
        if getattr(args, 'use_optimized_sb3_discrete_ppo', False):
            # OPTIMIZED VERSION: EAGLE-3 features + action caching + SB3 PPO
            enable_max_entropy = getattr(args, 'enable_max_entropy', True)
            max_entropy_inference_enabled = getattr(args, 'max_entropy_inference', True) and enable_max_entropy
            action_cache_steps = getattr(args, 'action_cache_steps', 10)
            hidden_size = getattr(args, 'hidden_size', 4096)
            policy_version = getattr(args, 'optimized_policy_version', 'standard')
            
            mode_description = "CUSTOM Max-Entropy Discrete PPO" if enable_max_entropy else "CUSTOM Standard Discrete PPO"
            version_description = f" ({policy_version.upper()} version)" if policy_version != "standard" else ""
            print(f"🚀⚡ Using {mode_description}{version_description} with EAGLE-3 features + action caching (every {action_cache_steps} steps)")
            
            # Parse network architecture based on policy version
            net_arch = parse_ppo_net_arch_args(getattr(args, 'ppo_net_arch', ''), policy_version)
            
            # Choose policy implementation based on version
            if policy_version == "ofl":
                PolicyClass = OptimizedSB3DiscretePPOOnlineTreePolicyOFL
                print(f"📋 Using OFL version with enhanced features (set_max_timesteps, set_training_mode, enhanced PPO updates)")
                print(f"📊 Network architecture: {net_arch}")
            else:
                PolicyClass = CustomPPOOnlineTreePolicy
                print(f"📋 Using custom PPO implementation (no SB3 dependency)")
                print(f"📊 Network architecture: {net_arch}")
            
            online_policy = PolicyClass(
                learning_rate=args.online_lr,
                n_steps=getattr(args, 'ppo_n_steps', 64),
                batch_size=getattr(args, 'ppo_batch_size', 32),
                n_epochs=getattr(args, 'ppo_epochs', 4),
                gamma=getattr(args, 'ppo_gamma', 0.95),
                gae_lambda=getattr(args, 'ppo_gae_lambda', 0.9),
                clip_range=getattr(args, 'ppo_clip_range', 0.2),
                vf_coef=getattr(args, 'ppo_vf_coef', 0.5),
                ent_coef=0.01,  # Standard entropy coefficient (will be overridden if max-entropy enabled)
                max_grad_norm=getattr(args, 'max_grad_norm', 0.5),
                # Max-entropy specific parameters
                enable_max_entropy=enable_max_entropy,
                max_entropy_ent_coef=getattr(args, 'max_entropy_ent_coef', 0.1),
                inference_temperature=getattr(args, 'inference_temperature', 1.5),
                max_entropy_inference=max_entropy_inference_enabled,
                # OPTIMIZATION parameters
                action_cache_steps=action_cache_steps,
                action_cache_enabled=getattr(args, 'action_cache_enabled', True),
                hidden_size=hidden_size,
                use_eagle3_features=getattr(args, 'use_eagle3_features', True),
                # NEW: Context-only state representation
                use_context_only_state=getattr(args, 'use_context_only_state', False),
                # NEW: Network architecture parameter
                net_arch=net_arch,
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=wandb_run_name,
                checkpoint_dir=checkpoint_dir,
                checkpoint_freq=getattr(args, 'checkpoint_freq', 100),
                max_checkpoints=getattr(args, 'max_checkpoints', 3)
            )
            reward_function = calculate_custom_ppo_reward
        elif getattr(args, 'use_optimized_dqn', False):
            # OPTIMIZED VERSION: EAGLE-3 features + action caching + DQN
            action_cache_steps = getattr(args, 'action_cache_steps', 10)
            hidden_size = getattr(args, 'hidden_size', 4096)
            
            print(f"🚀⚡ Using OPTIMIZED DQN with EAGLE-3 features + action caching (every {action_cache_steps} steps)")
            
            online_policy = OptimizedOnlineTreePolicy(
                learning_rate=args.online_lr,
                epsilon_start=args.online_epsilon_start,
                epsilon_end=args.online_epsilon_end,
                memory_size=args.online_memory_size,
                batch_size=args.online_batch_size,
                # OPTIMIZATION parameters
                action_cache_steps=action_cache_steps,
                action_cache_enabled=getattr(args, 'action_cache_enabled', True),
                hidden_size=hidden_size,
                use_eagle3_features=getattr(args, 'use_eagle3_features', True),
                # NEW: Context-only state representation
                use_context_only_state=getattr(args, 'use_context_only_state', False),
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=wandb_run_name,
                checkpoint_dir=checkpoint_dir,
                checkpoint_freq=getattr(args, 'checkpoint_freq', 100),
                max_checkpoints=getattr(args, 'max_checkpoints', 3)
            )
            reward_function = calculate_optimized_online_reward
        elif getattr(args, 'use_sb3_discrete_ppo', False):
            # Determine max-entropy mode settings (default: enabled)
            enable_max_entropy = getattr(args, 'enable_max_entropy', True)  # Default: True (max-entropy enabled)
            max_entropy_inference_enabled = getattr(args, 'max_entropy_inference', True) and enable_max_entropy  # Default: True if max-entropy enabled
            
            mode_description = "SB3 Max-Entropy Discrete PPO" if enable_max_entropy else "SB3 Standard Discrete PPO"
            print(f"🚀 Using {mode_description} for {'diverse' if enable_max_entropy else 'standard'} parameter exploration")
            
            online_policy = SB3DiscretePPOOnlineTreePolicy(
                learning_rate=args.online_lr,
                n_steps=getattr(args, 'ppo_n_steps', 64),
                batch_size=getattr(args, 'ppo_batch_size', 32),
                n_epochs=getattr(args, 'ppo_epochs', 4),
                gamma=getattr(args, 'ppo_gamma', 0.95),
                gae_lambda=getattr(args, 'ppo_gae_lambda', 0.9),
                clip_range=getattr(args, 'ppo_clip_range', 0.2),
                vf_coef=getattr(args, 'ppo_vf_coef', 0.5),
                ent_coef=0.01,  # Standard entropy coefficient (will be overridden if max-entropy enabled)
                max_grad_norm=getattr(args, 'max_grad_norm', 0.5),
                # Max-entropy specific parameters
                enable_max_entropy=enable_max_entropy,
                max_entropy_ent_coef=getattr(args, 'max_entropy_ent_coef', 0.1),
                inference_temperature=getattr(args, 'inference_temperature', 1.5),
                max_entropy_inference=max_entropy_inference_enabled,
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=wandb_run_name,
                checkpoint_dir=checkpoint_dir,
                checkpoint_freq=getattr(args, 'checkpoint_freq', 100),
                max_checkpoints=getattr(args, 'max_checkpoints', 3)
            )
            reward_function = calculate_sb3_discrete_ppo_reward
        elif getattr(args, 'use_discrete_ppo', False):
            print("🚀 Using Discrete PPO Algorithm for stable learning with parameter bins")
            online_policy = DiscretePPOOnlineTreePolicy(
                learning_rate=args.online_lr,
                ppo_epochs=getattr(args, 'ppo_epochs', 10),
                batch_size=getattr(args, 'ppo_batch_size', 64),
                clip_range=getattr(args, 'ppo_clip_range', 0.2),
                gamma=getattr(args, 'ppo_gamma', 0.99),
                gae_lambda=getattr(args, 'ppo_gae_lambda', 0.95),
                vf_coef=getattr(args, 'ppo_vf_coef', 0.5),
                max_grad_norm=getattr(args, 'max_grad_norm', 0.5),
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=wandb_run_name,
                checkpoint_dir=checkpoint_dir,
                checkpoint_freq=getattr(args, 'checkpoint_freq', 100),
                max_checkpoints=getattr(args, 'max_checkpoints', 3)
            )
            reward_function = calculate_discrete_ppo_reward
        elif getattr(args, 'use_ppo', False):
            print("🚀 Using PPO Algorithm for stable and efficient learning")
            online_policy = PPOOnlineTreePolicy(
                learning_rate=args.online_lr,
                n_steps=getattr(args, 'ppo_n_steps', 2048),
                batch_size=getattr(args, 'ppo_batch_size', 64),
                n_epochs=getattr(args, 'ppo_epochs', 10),
                gamma=getattr(args, 'ppo_gamma', 0.99),
                gae_lambda=getattr(args, 'ppo_gae_lambda', 0.95),
                clip_range=getattr(args, 'ppo_clip_range', 0.2),
                vf_coef=getattr(args, 'ppo_vf_coef', 0.5),
                max_grad_norm=getattr(args, 'max_grad_norm', 0.5),
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=wandb_run_name,
                checkpoint_dir=checkpoint_dir,
                checkpoint_freq=getattr(args, 'checkpoint_freq', 100),
                max_checkpoints=getattr(args, 'max_checkpoints', 3)
            )
            reward_function = calculate_ppo_online_reward
        elif getattr(args, 'continuous_action_space', False):
            print("🔄 Using Continuous Action Space (Actor-Critic) for maximum flexibility")
            online_policy = ContinuousOnlineTreePolicy(
                learning_rate=args.online_lr,
                epsilon_start=args.online_epsilon_start,
                epsilon_end=args.online_epsilon_end,
                memory_size=args.online_memory_size,
                batch_size=args.online_batch_size,
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=wandb_run_name,
                checkpoint_dir=checkpoint_dir,
                checkpoint_freq=getattr(args, 'checkpoint_freq', 100),
                max_checkpoints=getattr(args, 'max_checkpoints', 3)
            )
            reward_function = calculate_continuous_online_reward
        else:
            print("🎯 Using Discrete Action Space (116 valid combinations)")
            online_policy = OnlineTreePolicy(
                learning_rate=args.online_lr,
                epsilon_start=args.online_epsilon_start,
                epsilon_end=args.online_epsilon_end,
                memory_size=args.online_memory_size,
                batch_size=args.online_batch_size,
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=wandb_run_name,
                checkpoint_dir=checkpoint_dir,
                checkpoint_freq=getattr(args, 'checkpoint_freq', 100),
                max_checkpoints=getattr(args, 'max_checkpoints', 3)
            )
            reward_function = calculate_online_reward
        
        # Resume mechanism: check for existing checkpoint first, then explicit policy
        resumed = False
        if not args.online_inference_only:  # Only try to resume during training
            # Try to resume from latest checkpoint first (unless disabled)
            if getattr(args, 'resume_training', True) and not getattr(args, 'no_resume', False):
                if online_policy.load_checkpoint():
                    print("✅ Resumed training from latest checkpoint")
                    resumed = True
                    
                    # Set the training seed for reproducible continuation
                    if online_policy.training_seed:
                        training_seed = online_policy.training_seed
                        import random
                        random.seed(training_seed)
                        # Re-shuffle questions with same seed to get same order
                        random.shuffle(questions)
                        
                        # Skip already processed questions
                        questions_to_skip = online_policy.questions_processed
                        if questions_to_skip > 0 and questions_to_skip < len(questions):
                            questions = questions[questions_to_skip:]
                            print(f"📋 Resuming from question {questions_to_skip+1}/{len(questions) + questions_to_skip}")
                        elif questions_to_skip >= len(questions):
                            print("⚠️  All questions already processed in checkpoint")
                            print(f"processed {questions_to_skip} questions")
                            questions = []
                            exit(0)
                    else:
                        print("⚠️  No training seed in checkpoint, shuffling with default seed")
                        training_seed = getattr(args, 'training_seed', 42)
                        online_policy.set_training_seed(training_seed)
                        random.shuffle(questions)
        
        # Fallback to explicit policy path if no checkpoint resume
        if not resumed:
            if args.online_policy_path:
                # For SB3 discrete PPO, check the converted path format
                policy_exists = False
                if getattr(args, 'use_sb3_discrete_ppo', False):
                    # Check for SB3 format: .pth -> _sb3.zip
                    sb3_path = args.online_policy_path.replace('.pth', '_sb3.zip') if args.online_policy_path.endswith('.pth') else args.online_policy_path + '.zip'
                    policy_exists = os.path.exists(sb3_path)
                    if policy_exists:
                        print(f"Found SB3 policy file: {sb3_path}")
                else:
                    # Standard path check for other policies
                    policy_exists = os.path.exists(args.online_policy_path)
                
                if policy_exists:
                    # Load the policy (let the policy class handle path conversion internally)
                    if online_policy.load(args.online_policy_path):
                        print(f"Loaded existing online policy from {args.online_policy_path}")
                    else:
                        print("❗ Failed to load online policy: file exists but loading failed")
                        if args.online_inference_only:
                            print("❗ Online inference mode only, no training allowed")
                            quit()
                else:
                    print(f"Policy file not found: {args.online_policy_path}")
                    if args.online_inference_only:
                        print("❗ Failed to load online policy: online inference mode only, no training allowed")
                        quit()
                    else:
                        print("Starting with fresh online policy")
            else:
                print("Starting with fresh online policy")
                if args.online_inference_only:
                    print("❗ Failed to load online policy: online inference mode only, no training allowed")
                    quit()
                
            # Set training seed for fresh start
            if not args.online_inference_only:
                training_seed = getattr(args, 'training_seed', 42)
                online_policy.set_training_seed(training_seed)
                # Questions already shuffled above with this seed
            
    elif args.use_rl_policy:
        # Use traditional offline RL policy
        try:
            rl_policy = RLTreePolicy(args.rl_policy_path)
            print(f"Loaded offline RL policy from {args.rl_policy_path}")
        except Exception as e:
            print(f"Failed to load RL policy: {e}")
            print("Falling back to fixed parameters from args")
            rl_policy = None

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({
                "role": "user",
                "content": qs
            })
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer([prompt],add_special_tokens=False,).input_ids

            # Get parameters from policy
            full_context = " ".join([msg["content"] for msg in messages])
            
            if online_policy is not None:
                # Check if this is an optimized policy
                is_optimized_policy = hasattr(online_policy, 'use_eagle3_features') and online_policy.use_eagle3_features
                
                if is_optimized_policy:
                    # Optimized RL: predict parameters using EAGLE-3 features (inference mode during warmup)
                    # Note: hidden_states not available during warmup, so pass None
                    # predicted_total_tokens, predicted_depth, predicted_top_k = online_policy.predict_parameters(
                        # context=full_context, hidden_states=None, training_mode=False  # No exploration during warmup
                    # )
                    predicted_total_tokens, predicted_depth, predicted_top_k = (96, 8, 20)
                else:
                    # Traditional Online RL: predict parameters (inference mode during warmup)
                    predicted_total_tokens, predicted_depth, predicted_top_k = online_policy.predict_parameters(
                        full_context, training_mode=False  # No exploration during warmup
                    )
                print(f"Online RL predicted params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
            elif rl_policy is not None:
                # Offline RL policy
                predicted_total_tokens, predicted_depth, predicted_top_k = rl_policy.predict_parameters(full_context)
                print(f"Offline RL predicted params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
            else:
                # Fallback to defaults
                predicted_total_tokens = args.total_token
                predicted_depth = args.depth
                predicted_top_k = args.top_k
                # print(f"Using default params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
            # Clear GPU memory before retry
            # if torch.cuda.is_available():
                # torch.cuda.empty_cache()         
            torch.cuda.synchronize()
            start_time = time.time()

            # Use no_grad for warmup since it's always inference mode
            try:
                with torch.no_grad():
                    if args.use_stepwise_rl and online_policy is not None:
                        # Step-wise RL mode: pass RL policy to eagenerate
                        result = model.eagenerate(
                            torch.as_tensor(input_ids).cuda(),
                            temperature=temperature,
                            log=True,
                            is_llama3=True,
                            total_tokens=predicted_total_tokens,  # Fallback values
                            depth=predicted_depth,
                            tree_top_k=predicted_top_k,
                            rl_policy=online_policy,
                            training_mode=False,  # No training during warmup
                            max_length=max_length_param,
                        )
                        # Handle variable return values from step-wise RL
                        if len(result) == 5:  # Step-wise RL with log=True
                            output_ids, new_token, idx, step_rewards, step_count = result
                            print(f"Warmup with step-wise RL completed: {new_token} tokens, {step_count} steps")
                        else:  # Fallback to traditional
                            output_ids, new_token, idx = result
                            print(f"Warmup completed: {new_token} tokens generated")
                    else:
                        # Traditional mode: fixed parameters for entire generation
                        output_ids, new_token, idx = model.eagenerate(
                            torch.as_tensor(input_ids).cuda(),
                            temperature=temperature,
                            log=True,
                            is_llama3=True,
                            total_tokens=predicted_total_tokens,
                            depth=predicted_depth,
                            tree_top_k=predicted_top_k,
                            max_length=max_length_param,
                        )
            except RuntimeError as e:
                if ("selected index k out of range" in str(e) or "exceeds dimension size" in str(e) or 
                    "start" in str(e) or "KV cache buffer overflow" in str(e) or 
                    "CUDA out of memory" in str(e) or "out of memory" in str(e)):
                    print(f"❌ Warmup error with params: tt={predicted_total_tokens}, d={predicted_depth}, k={predicted_top_k}")
                    print(f"   Error: {e}")
                    print(f"   Falling back to ultra-conservative warmup parameters...")
                    
                    # Use ultra-safe parameters for warmup that definitely won't overflow
                    safe_total_tokens = 60
                    safe_depth = 5
                    safe_top_k = 10
                
                    try:
                        with torch.no_grad():
                            if args.use_stepwise_rl and online_policy is not None:
                                # Step-wise RL fallback mode
                                result = model.eagenerate(
                                    torch.as_tensor(input_ids).cuda(),
                                    temperature=temperature,
                                    log=True,
                                    is_llama3=True,
                                    total_tokens=safe_total_tokens,  # Conservative fallback
                                    depth=safe_depth,
                                    tree_top_k=safe_top_k,
                                    rl_policy=online_policy,
                                    training_mode=False,  # No training during fallback
                                    max_length=max_length_param,
                                )
                                # Handle variable return values from step-wise RL
                                if len(result) == 5:  # Step-wise RL with log=True
                                    output_ids, new_token, idx, step_rewards, step_count = result
                                    print(f"Fallback warmup with step-wise RL: {new_token} tokens, {step_count} steps")
                                else:  # Fallback to traditional
                                    output_ids, new_token, idx = result
                            else:
                                # Traditional fallback mode  
                                output_ids, new_token, idx = model.eagenerate(
                                    torch.as_tensor(input_ids).cuda(),
                                    temperature=temperature,
                                    log=True,
                                    is_llama3=True,
                                    total_tokens=safe_total_tokens,
                                    depth=safe_depth,
                                    tree_top_k=safe_top_k,
                                    max_length=max_length_param,
                                )
                    except RuntimeError as e2:
                        print(f"❌ Even ultra-conservative warmup failed: {e2}")
                        print("   Skipping warmup - proceeding with standard generation")
                        # We'll skip warmup if even ultra-conservative parameters fail
                        # Set dummy values to avoid UnboundLocalError
                        # Get device from model instead of input_ids (which is a list)
                        device = next(model.parameters()).device
                        output_ids = torch.tensor([[tokenizer.eos_token_id]]).to(device)
                        new_token = torch.tensor(0).to(device)
                        idx = torch.tensor(0).to(device)
                        total_time = 0.0
                else:
                    raise e  # Re-raise if it's a different error
            
            # Only process output_ids if warmup was successful
            if 'output_ids' in locals() and output_ids is not None:
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
                
                # be consistent with the template's stop_token_ids
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                if stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                # stop_str = "</s>"
                # if stop_str and output.find(stop_str) > 0:
                #     output = output[: output.find(stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({
                    "role": "assistant",
                    "content": output
                })
            else:
                # Warmup failed completely, add empty response
                print("   ⚠️  Warmup failed - adding empty response")
                turns.append("")
                idxs.append(0)
                new_tokens.append(0)
                wall_time.append(0.0)
                messages.append({
                    "role": "assistant",
                    "content": ""
                })
    print('Warmup done')

    # questions=questions[6:]
    question_count = 0
    for question in tqdm(questions, desc="Processing questions"):
        
        # Track question processing for online RL resume mechanism (count even skipped questions)
        question_count += 1
        if online_policy is not None and not args.online_inference_only:
            online_policy.increment_questions_processed()
            
            # Progress update with checkpoint info
            if question_count % 10 == 0:
                resume_info = online_policy.get_resume_info()
                step_count = resume_info.get('step_count', 0)
                progress_msg = (f"📊 Progress: {question_count}/{len(questions)} questions, "
                               f"Step: {step_count}")
                
                # Add policy-specific metrics
                if 'epsilon' in resume_info:
                    # DQN-based policies (discrete/continuous Actor-Critic)
                    progress_msg += f", Epsilon: {resume_info['epsilon']:.3f}"
                elif 'ppo_updates' in resume_info:
                    # PPO-based policies
                    progress_msg += f", PPO Updates: {resume_info['ppo_updates']}"
                
                print(progress_msg)
                
                if online_policy.should_save_checkpoint():
                    print(f"   💾 Checkpoint will be saved at step {step_count}")

        choices = []
        question_failed = False  # Track if any turn in this question failed
        
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({
                    "role": "user",
                    "content": qs
                })
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids

                # Get parameters from policy
                full_context = " ".join([msg["content"] for msg in messages])
                
                if online_policy is not None:
                    # Check if this is an optimized policy
                    is_optimized_policy = hasattr(online_policy, 'use_eagle3_features') and online_policy.use_eagle3_features
                    
                    if is_optimized_policy:
                        # Optimized RL: predict parameters with appropriate training mode
                        if args.online_inference_only:
                            # Inference-only mode: no training, no exploration
                            # predicted_total_tokens, predicted_depth, predicted_top_k = online_policy.predict_parameters(
                                # context=full_context, hidden_states=None, training_mode=False
                            # )
                            predicted_total_tokens, predicted_depth, predicted_top_k = (96, 8, 20)
                            # print(f"Optimized Online RL inference params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
                        else:
                            # Training mode: enable exploration and learning
                            # predicted_total_tokens, predicted_depth, predicted_top_k = online_policy.predict_parameters(
                                # context=full_context, hidden_states=None, training_mode=True
                            # )
                            predicted_total_tokens, predicted_depth, predicted_top_k = (96, 8, 20)
                            print(f"Optimized Online RL training params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
                    else:
                        # Traditional Online RL: predict parameters with appropriate training mode
                        if args.online_inference_only:
                            # Inference-only mode: no training, no exploration
                            predicted_total_tokens, predicted_depth, predicted_top_k = online_policy.predict_parameters(
                                full_context, training_mode=False
                            )
                            # print(f"Online RL inference params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
                        else:
                            # Training mode: enable exploration and learning
                            predicted_total_tokens, predicted_depth, predicted_top_k = online_policy.predict_parameters(
                                full_context, training_mode=True
                            )
                            print(f"Online RL training params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
                elif rl_policy is not None:
                    # Offline RL policy
                    predicted_total_tokens, predicted_depth, predicted_top_k = rl_policy.predict_parameters(full_context)
                    print(f"Offline RL predicted params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")
                else:
                    # Fallback to defaults
                    predicted_total_tokens = args.total_token
                    predicted_depth = args.depth
                    predicted_top_k = args.top_k
                    # print(f"Using default params: total_tokens={predicted_total_tokens}, depth={predicted_depth}, top_k={predicted_top_k}")

                # Clear GPU memory before retry
                # if torch.cuda.is_available():
                    # torch.cuda.empty_cache()
                torch.cuda.synchronize()
                start_time = time.time()

                # Use no_grad for model inference to save memory and computation
                max_retries = 1 # default is 3, here, we set it to 1 for faster inference
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        # Conditionally use torch.no_grad() only during inference mode
                        training_mode = not args.online_inference_only
                        
                        if training_mode and online_policy is not None:
                            # Training mode: allow gradients for RL policy learning
                            if args.use_stepwise_rl and online_policy is not None:
                                # Step-wise RL mode: pass RL policy and enable training
                                result = model.eagenerate(
                                    torch.as_tensor(input_ids).cuda(),
                                    temperature=temperature,
                                    log=True,
                                    is_llama3=True,
                                    total_tokens=predicted_total_tokens,  # Fallback values
                                    depth=predicted_depth,
                                    tree_top_k=predicted_top_k,
                                    rl_policy=online_policy,
                                    training_mode=training_mode,
                                    max_length=max_length_param,
                                )
                                # Handle variable return values from step-wise RL
                                if len(result) == 5:  # Step-wise RL with log=True
                                    output_ids, new_token, idx, step_rewards, step_count = result
                                    if training_mode:
                                        print(f"Step-wise RL training: {new_token} tokens, {step_count} steps, avg reward: {sum(step_rewards)/len(step_rewards):.2f}")
                                    else:
                                        # print(f"Step-wise RL inference: {new_token} tokens, {step_count} steps")
                                        pass
                                else:  # Fallback to traditional
                                    output_ids, new_token, idx = result
                                    print(f"Step-wise RL (fallback): {new_token} tokens, {idx+1} steps")
                            else:
                                # Traditional mode: fixed parameters for entire generation
                                output_ids, new_token, idx = model.eagenerate(
                                    torch.as_tensor(input_ids).cuda(),
                                    temperature=temperature,
                                    log=True,
                                    is_llama3=True,
                                    total_tokens=predicted_total_tokens,
                                    depth=predicted_depth,
                                    tree_top_k=predicted_top_k,
                                    max_length=max_length_param,
                                )
                        else:
                            # Inference mode: use torch.no_grad() for memory efficiency
                            with torch.no_grad():
                                if args.use_stepwise_rl and online_policy is not None:
                                    # Step-wise RL mode: pass RL policy but disable training
                                    result = model.eagenerate(
                                        torch.as_tensor(input_ids).cuda(),
                                        temperature=temperature,
                                        log=True,
                                        is_llama3=True,
                                        total_tokens=predicted_total_tokens,  # Fallback values
                                        depth=predicted_depth,
                                        tree_top_k=predicted_top_k,
                                        rl_policy=online_policy,
                                        training_mode=False,  # Inference only
                                        max_length=max_length_param,
                                    )
                                    # Handle variable return values from step-wise RL
                                    if len(result) == 5:  # Step-wise RL with log=True
                                        output_ids, new_token, idx, step_rewards, step_count = result
                                        # print(f"Step-wise RL inference: {new_token} tokens, {step_count} steps")
                                        pass
                                    else:  # Fallback to traditional
                                        output_ids, new_token, idx = result
                                        print(f"Step-wise RL (fallback): {new_token} tokens, {idx+1} steps")
                                else:
                                    # Traditional mode: fixed parameters for entire generation
                                    output_ids, new_token, idx = model.eagenerate(
                                        torch.as_tensor(input_ids).cuda(),
                                        temperature=temperature,
                                        log=True,
                                        is_llama3=True,
                                        total_tokens=predicted_total_tokens,
                                        depth=predicted_depth,
                                        tree_top_k=predicted_top_k,
                                        max_length=max_length_param,
                                    )
                        success = True
                        
                    except RuntimeError as e:
                        # raise e
                        if ("selected index k out of range" in str(e) or "exceeds dimension size" in str(e) or 
                            "start" in str(e) or "KV cache buffer overflow" in str(e) or 
                            "CUDA out of memory" in str(e) or "out of memory" in str(e)):
                            retry_count += 1
                            print(f"❌ Runtime error with params: tt={predicted_total_tokens}, d={predicted_depth}, k={predicted_top_k} (attempt {retry_count}/{max_retries})")
                            print(f"   Error: {e}")
                            
                            if retry_count < max_retries:
                                print(f"   Trying more conservative parameters...")
                                # Make parameters increasingly conservative with each retry
                                if "KV cache buffer overflow" in str(e) or "CUDA out of memory" in str(e) or "out of memory" in str(e):
                                    # For memory-related errors, be very aggressive with reduction
                                    if retry_count == 1:
                                        predicted_total_tokens = 60
                                        predicted_depth = 5
                                        predicted_top_k = 10
                                    elif retry_count == 2:
                                        predicted_total_tokens = 8
                                        predicted_depth = 2
                                        predicted_top_k = 2
                                else:
                                    # For other errors, use moderate reduction
                                    if retry_count == 1:
                                        # First retry: moderate reduction
                                        predicted_total_tokens = 60
                                        predicted_depth = 5
                                        predicted_top_k = 10
                                    elif retry_count == 2:
                                        # Second retry: very conservative
                                        predicted_total_tokens = 16
                                        predicted_depth = 3
                                        predicted_top_k = 4
                                
                                print(f"   Using safer params: tt={predicted_total_tokens}, d={predicted_depth}, k={predicted_top_k}")
                            else:
                                print(f"   All retries failed. Skipping this question...")
                                break
                        else:
                            # Re-raise non-KV cache related errors
                            raise e
                
                # If all retries failed, skip this entire question
                if not success:
                    print(f"⏭️  Skipping question {question_count}/{len(questions)} due to persistent KV cache errors")
                    question_failed = True
                    break  # Break out of the turns loop
                    
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's stop_token_ids
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                if stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                # stop_str = "</s>"
                # if stop_str and output.find(stop_str) > 0:
                #     output = output[: output.find(stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                # Online RL: Update policy with reward from this generation (if training enabled)
                # DISABLED: Using step-wise RL only - question-level updates are handled in ea_model.py
                if not args.use_stepwise_rl and online_policy is not None and not args.online_inference_only:
                    # Convert tensor values to Python scalars
                    new_token_scalar = int(new_token.cpu()) if hasattr(new_token, 'cpu') else int(new_token)
                    
                    # Calculate reward for online learning (use appropriate reward function)
                    online_reward = reward_function(
                        total_time, new_token_scalar, predicted_total_tokens, 
                        predicted_depth, predicted_top_k
                    )
                    
                    # DISABLED: This would duplicate step-wise RL updates from ea_model.py
                    online_policy.update_policy(online_reward, total_time, new_token_scalar)
                    # DISABLED: Progress logging - step-wise RL handles its own logging
                    # Print learning progress periodically
                    if not args.use_stepwise_rl and not args.online_inference_only and online_policy.step_count % 20 == 0:
                        stats = online_policy.get_performance_stats()
                        print(f"Online RL Update: Reward={online_reward:.3f}, "
                              f"Avg Recent Reward={stats.get('avg_reward_recent', 0):.3f}, "
                              f"Tokens/sec={new_token_scalar/total_time:.1f}")
                        
                        # Additional wandb logging for progress tracking
                        if online_policy.use_wandb:
                            wandb.log({
                                "progress_reward": online_reward,
                                "progress_tokens_per_sec": new_token_scalar/total_time,
                                "progress_step": online_policy.step_count
                            })

                # Collect RL training data if requested
                if args.collect_rl_data:
                    # Convert tensor values to Python scalars
                    new_token_scalar = int(new_token.cpu()) if hasattr(new_token, 'cpu') else int(new_token)
                    rl_reward = calculate_real_reward(
                        total_time, new_token_scalar, predicted_total_tokens, 
                        predicted_depth, predicted_top_k
                    )
                    rl_data_entry = {
                        "question_id": question["question_id"],
                        "turn": j,
                        "choice": i,
                        "context": " ".join([msg["content"] for msg in messages[:-1]]),  # Exclude assistant response
                        "total_tokens": predicted_total_tokens,
                        "depth": predicted_depth,
                        "top_k": predicted_top_k,
                        "generation_time": total_time,
                        "new_tokens": new_token_scalar,
                        "reward": rl_reward,
                        "tokens_per_second": new_token_scalar / total_time if total_time > 0 else 0
                    }
                    rl_data_entries.append(rl_data_entry)

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({
                    "role": "assistant",
                    "content": output
                })
            # torch.cuda.empty_cache()
            if question_failed:
                break  # Break out of choices loop if any turn failed
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

        # Only save answer if question was successfully processed
        if not question_failed:
            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")
        else:
            print(f"   Question {question_count} skipped - not saved to answer file")

    # Save online RL policy if used and training was enabled
    if online_policy is not None:
        if args.online_inference_only:
            # Inference-only mode: just show statistics
            final_stats = online_policy.get_performance_stats()
            print(f"\n=== Online RL Inference Complete ===")
            print(f"Used pre-trained policy for parameter optimization")
            print(f"Questions processed: {len(questions)} (original order)")
            if final_stats.get('most_used_params'):
                print(f"Most used parameters: {final_stats.get('most_used_params', [])}")
        else:
            # Training mode: save updated policy with enhanced statistics
            save_path = args.online_policy_save_path or "online_tree_policy_trained.pth"
            online_policy.save(save_path)
            online_policy.save_checkpoint()
            
            # Print comprehensive training statistics
            final_stats = online_policy.get_performance_stats()
            print(f"\n=== Online RL Training Complete ===")
            print(f"Questions processed: {len(questions)} (repeated & shuffled for training)")
            print(f"Total episodes: {final_stats.get('total_episodes', 0)}")
            print(f"Final average reward: {final_stats.get('avg_reward_recent', 0):.4f}")
            print(f"Most used parameters: {final_stats.get('most_used_params', [])}")
            print(f"Policy saved to: {save_path}")
            
            # Show parameter exploration diversity
            if final_stats.get('total_episodes', 0) > 0:
                param_count = len(final_stats.get('most_used_params', []))
                print(f"Parameter combinations explored: {param_count}")
                # Final wandb summary
                if online_policy.use_wandb:
                    summary_data = {
                        "final_avg_reward": final_stats.get('avg_reward_recent', 0),
                        "total_episodes": final_stats.get('total_episodes', 0),
                        "parameter_combinations_used": param_count,
                        "questions_processed": len(questions),
                    }
                    
                    # Add policy-specific metrics
                    if hasattr(online_policy, 'epsilon'):
                        # DQN-based policies (discrete/continuous Actor-Critic)
                        summary_data["final_epsilon"] = online_policy.epsilon
                    elif hasattr(online_policy, 'ppo_updates'):
                        # PPO-based policies
                        summary_data["ppo_updates"] = online_policy.ppo_updates
                        summary_data["policy_type"] = "ppo"
                    
                    wandb.run.summary.update(summary_data)
                    wandb.finish()
            
            # Print final statistics
            final_stats = online_policy.get_performance_stats()
            print(f"\n=== Online RL Training Complete ===")
            print(f"Total episodes: {final_stats.get('total_episodes', 0)}")
            print(f"Final average reward: {final_stats.get('avg_reward_recent', 0):.4f}")
            print(f"Most used parameters: {final_stats.get('most_used_params', [])}")
            print(f"Policy saved to: {save_path}")

    # Save RL training data if collected
    if args.collect_rl_data and rl_data_entries:
        rl_data_file = os.path.expanduser(args.rl_data_file)
        rl_data_dir = os.path.dirname(rl_data_file)
        if rl_data_dir:  # Only create directory if there's a directory path
            os.makedirs(rl_data_dir, exist_ok=True)
        with open(rl_data_file, "a") as fout:
            for entry in rl_data_entries:
                fout.write(json.dumps(entry) + "\n")
        print(f"Saved {len(rl_data_entries)} RL training entries to {rl_data_file}")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/home/lyh/weights/hf/eagle3/llama31chat/8B/",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama31chat/8B/",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="llama38b2_40")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        help="Custom question file path (overrides --bench-name). Use this for training with custom datasets like the combined conversational data."
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="total-token = The total number of drafted tokens in the tree + 1. Used as fallback when RL policy is disabled or fails.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="depth = The maximum number of draft length - 1. Used as fallback when RL policy is disabled or fails.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of drafted tokens in each layer. Used as fallback when RL policy is disabled or fails.",
    )

    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument(
        "--use-eagle3", "--use_eagle3",
        dest="use_eagle3",
        action="store_true"
    )
    parser.add_argument(
        "--use-rl-policy",
        action="store_true",
        help="Use RL policy to predict tree parameters dynamically"
    )
    parser.add_argument(
        "--rl-policy-path",
        type=str,
        default="ppo_tree_policy_discrete.zip",
        help="Path to trained RL policy model"
    )
    parser.add_argument(
        "--collect-rl-data",
        action="store_true",
        help="Collect data for RL training (save generation metrics)"
    )
    parser.add_argument(
        "--rl-data-file",
        type=str,
        default="rl_training_data.jsonl",
        help="File to save RL training data"
    )
    
    # Online RL arguments
    parser.add_argument(
        "--use-online-rl",
        action="store_true",
        help="Use online RL for real-time parameter optimization (training mode repeats & shuffles questions)"
    )
    parser.add_argument(
        "--online-policy-path",
        type=str,
        help="Path to load existing online RL policy (optional)"
    )
    parser.add_argument(
        "--online-policy-save-path",
        type=str,
        default="online_tree_policy_trained.pth",
        help="Path to save trained online RL policy"
    )
    parser.add_argument(
        "--online-lr",
        type=float,
        default=3e-4,
        help="Learning rate for online RL policy"
    )
    parser.add_argument(
        "--online-epsilon-start",
        type=float,
        default=0.9,
        help="Initial exploration rate for online RL"
    )
    parser.add_argument(
        "--online-epsilon-end",
        type=float,
        default=0.05,
        help="Final exploration rate for online RL"
    )
    parser.add_argument(
        "--online-memory-size",
        type=int,
        default=1000,
        help="Experience replay memory size for online RL"
    )
    parser.add_argument(
        "--online-batch-size",
        type=int,
        default=32,
        help="Batch size for online RL training"
    )
    parser.add_argument(
        "--online-inference-only",
        action="store_true",
        help="Use online RL policy for inference only (no training/learning)"
    )
    parser.add_argument(
        "--online-repeat-factor",
        type=int,
        default=1,
        help="Number of times to repeat questions during online RL training (inference mode ignores this)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="eagle-online-rl",
        help="Wandb project name for logging (only used during online RL training)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging even during training"
    )
    parser.add_argument(
        "--force-disable-wandb-multi-gpu",
        action="store_true",
        help="[DEPRECATED] Force disable wandb on multi-GPU setups. Now wandb is attempted with timeout on all setups."
    )
    parser.add_argument(
        "--wandb-timeout",
        type=int,
        default=10,
        help="Timeout in seconds for wandb authentication (default: 10s). Prevents hanging on DNS/network issues."
    )
    
    # Resume and checkpoint arguments
    parser.add_argument(
        "--training-seed",
        type=int,
        default=42,
        help="Random seed for reproducible training data shuffling (needed for resume)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save training checkpoints for resume capability"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100,
        help="Save checkpoint every N training steps"
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=1,
        help="Maximum number of checkpoints to keep (older ones are automatically deleted)"
    )
    parser.add_argument(
        "--resume-training",
        type=bool,
        default=True,
        help="Automatically resume from latest checkpoint if available (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic resume and start fresh training"
    )
    parser.add_argument(
        "--continuous-action-space",
        action="store_true",
        help="Use continuous action space instead of discrete bins for more flexibility"
    )
    
    # PPO-specific arguments
    parser.add_argument(
        "--use-ppo",
        action="store_true",
        help="Use PPO (Proximal Policy Optimization) algorithm for stable continuous learning"
    )
    parser.add_argument(
        "--use-discrete-ppo",
        action="store_true",
        help="Use PPO with discrete action space (combines PPO stability with parameter bins)"
    )
    parser.add_argument(
        "--use-sb3-discrete-ppo",
        action="store_true",
        help="Use Stable Baselines 3 Discrete PPO for optimized and robust learning"
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=10,
        help="Number of epochs when optimizing the surrogate loss for PPO (both continuous and discrete)"
    )
    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=64,
        help="Minibatch size for PPO updates (both continuous and discrete)"
    )
    parser.add_argument(
        "--ppo-clip-range",
        type=float,
        default=0.2,
        help="Clipping parameter for PPO surrogate loss (both continuous and discrete)"
    )
    parser.add_argument(
        "--ppo-gamma",
        type=float,
        default=0.99,
        help="Discount factor for PPO (both continuous and discrete)"
    )
    parser.add_argument(
        "--ppo-gae-lambda",
        type=float,
        default=0.95,
        help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator (both continuous and discrete)"
    )
    parser.add_argument(
        "--ppo-n-steps",
        type=int,
        default=2048,
        help="Number of steps to run for each environment per update for PPO (continuous PPO only)"
    )
    parser.add_argument(
        "--ppo-ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for the loss calculation (SB3 PPO only)"
    )
    parser.add_argument(
        "--ppo-vf-coef",
        type=float,
        default=0.5,
        help="Value function coefficient for the loss calculation (SB3 PPO only)"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="The maximum value for the gradient clipping (PPO only)"
    )
    
    # Max-entropy RL arguments
    parser.add_argument(
        "--enable-max-entropy",
        action="store_true",
        default=True,  # DEFAULT: Max-entropy mode enabled
        help="Enable max-entropy RL mode for SB3 PPO (higher exploration, temperature-based inference) - DEFAULT ENABLED"
    )
    parser.add_argument(
        "--disable-max-entropy",
        action="store_true",
        help="Disable max-entropy RL mode and use standard PPO (lower exploration, deterministic inference)"
    )
    parser.add_argument(
        "--max-entropy-ent-coef",
        type=float,
        default=0.1,
        help="Entropy coefficient for max-entropy mode (default: 0.1, standard PPO uses ~0.01)"
    )
    parser.add_argument(
        "--inference-temperature",
        type=float,
        default=1.5,
        help="Temperature for exploration during inference (Max-Entropy mode only). Higher values increase diversity."
    )
    parser.add_argument(
        "--max-entropy-inference",
        action="store_true",
        default=True,  # DEFAULT: Max-entropy inference enabled
        help="Enable max-entropy inference for parameter diversity during inference (DEFAULT ENABLED)"
    )
    parser.add_argument(
        "--no-max-entropy-inference",
        action="store_true",
        help="Disable max-entropy inference and use deterministic inference (SB3 PPO only)"
    )
    
    # Step-wise RL arguments
    parser.add_argument(
        "--use-stepwise-rl",
        action="store_true",
        help="Enable step-wise RL: predict parameters at each draft step instead of once per turn (requires --use-online-rl)"
    )
    parser.add_argument(
        "--stepwise-reward-type",
        type=str,
        default="tokens_per_second",
        choices=["tokens_per_second", "acceptance_rate", "combined"],
        help="Type of reward to use for step-wise RL training (step-wise RL only)"
    )
    
    # Optimized RL arguments - Layer Features + Action Caching
    parser.add_argument(
        "--use-optimized-rl",
        action="store_true",
        help="Use optimized RL policy with layer feature concatenation and reduced action frequency"
    )
    parser.add_argument(
        "--use-optimized-sb3-discrete-ppo",
        action="store_true",
        help="Use optimized SB3 Discrete PPO with EAGLE-3 features and action caching"
    )
    parser.add_argument(
        "--optimized-policy-version",
        type=str,
        choices=["standard", "ofl"],
        default="standard",
        help="Choose between standard optimized policy or OFL (Offline) version with enhanced features"
    )
    parser.add_argument(
        "--use-optimized-dqn",
        action="store_true",
        help="Use optimized DQN with EAGLE-3 features and action caching"
    )
    parser.add_argument(
        "--action-cache-steps",
        type=int,
        default=10,
        help="Generate action every N steps for action caching optimization (optimized policies only)"
    )
    parser.add_argument(
        "--action-cache-enabled",
        action="store_true",
        default=True,
        help="Enable action caching optimization (optimized policies only)"
    )
    parser.add_argument(
        "--use-eagle3-features",
        action="store_true",
        default=True,
        help="Use EAGLE-3 layer features instead of SBERT text embeddings (optimized policies only)"
    )
    parser.add_argument(
        "--use-context-only-state",
        action="store_true",
        default=False,
        help="NEW: Use SBERT context embeddings directly (384D) instead of EAGLE-3 features or projection. Overrides --use-eagle3-features when enabled."
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden size for EAGLE-3 layer features (typically 4096 for LLaMA models)"
    )

    parser.add_argument(
        "--ppo-net-arch",
        type=str,
        default="",
        help="Network architecture for PPO policies. For standard version: comma-separated integers (e.g., '512,256,128'). For OFL version: '64,64' for same pi/vf or '64,64;128,128' for different pi/vf"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model data type"
    )

    args = parser.parse_args()
    

    # Handle max-entropy mode defaults and overrides
    if getattr(args, 'disable_max_entropy', False):
        args.enable_max_entropy = False
        args.max_entropy_inference = False
        print("🔧 Max-entropy mode disabled via --disable-max-entropy flag")
    else:
        # Default to max-entropy mode unless explicitly disabled
        if not hasattr(args, 'enable_max_entropy'):
            args.enable_max_entropy = True
        if not hasattr(args, 'max_entropy_inference'):
            args.max_entropy_inference = True
    
    # Handle max-entropy inference argument
    if getattr(args, 'no_max_entropy_inference', False):
        args.max_entropy_inference = False
    
    # Validate step-wise RL arguments
    if getattr(args, 'use_stepwise_rl', False):
        if not args.use_online_rl:
            print("❌ Error: --use-stepwise-rl requires --use-online-rl")
            print("   Step-wise RL can only be used with online RL policies that support real-time learning")
            exit(1)
        print("🔀 Step-wise RL enabled: Parameters will be predicted at each draft step")
        print(f"   Reward type: {getattr(args, 'stepwise_reward_type', 'tokens_per_second')}")

    for k,v in vars(args).items():
        print(f"{k}={v}")

    # Print helpful information about RL vs fixed parameters
    if args.use_online_rl:
        if getattr(args, 'use_sb3_discrete_ppo', False):
            enable_max_entropy = getattr(args, 'enable_max_entropy', True)  # Default: True (max-entropy enabled)
            mode_name = "SB3 Max-Entropy Discrete PPO" if enable_max_entropy else "SB3 Standard Discrete PPO"
            action_space_type = f"{mode_name} ({'Diverse Exploration' if enable_max_entropy else 'Standard Exploration'})"
        elif getattr(args, 'use_discrete_ppo', False):
            action_space_type = "Discrete PPO (Actor-Critic)"
        elif getattr(args, 'use_ppo', False):
            action_space_type = "PPO (Continuous)"
        elif getattr(args, 'continuous_action_space', False):
            action_space_type = "Continuous (Actor-Critic)"
        else:
            action_space_type = "Discrete (DQN)"
            
        print(f"\n🚀 Online RL Mode: Real-time learning and parameter optimization ({action_space_type})")
        
        # Step-wise RL information
        if getattr(args, 'use_stepwise_rl', False):
            print(f"🔀 Step-wise RL: Parameters predicted at each draft step (vs once per turn)")
            print(f"   Reward type: {getattr(args, 'stepwise_reward_type', 'tokens_per_second')}")
            print(f"   Training granularity: Sub-turn level (fine-grained optimization)")
        else:
            print(f"🎯 Turn-wise RL: Parameters predicted once per turn (traditional)")
            print(f"   Training granularity: Turn level (coarse-grained optimization)")
        
        print(f"   Learning rate: {args.online_lr}")
        
        if getattr(args, 'use_sb3_discrete_ppo', False):
            enable_max_entropy = getattr(args, 'enable_max_entropy', True)  # Default: True (max-entropy enabled)
            print(f"   SB3 Mode: {'Max-Entropy' if enable_max_entropy else 'Standard'} PPO")
            print(f"   SB3 PPO n_steps: {getattr(args, 'ppo_n_steps', 64)}")
            print(f"   SB3 PPO batch_size: {getattr(args, 'ppo_batch_size', 32)}")
            print(f"   SB3 PPO n_epochs: {getattr(args, 'ppo_epochs', 4)}")
            print(f"   SB3 PPO gamma: {getattr(args, 'ppo_gamma', 0.95)}")
            print(f"   SB3 PPO GAE lambda: {getattr(args, 'ppo_gae_lambda', 0.9)}")
            
            if enable_max_entropy:
                print(f"   Entropy coefficient: {getattr(args, 'max_entropy_ent_coef', 0.1)} (HIGH for max-entropy)")
                print(f"   Inference temperature: {getattr(args, 'inference_temperature', 1.5)} (for exploration during inference)")
                print(f"   Max-entropy inference: {getattr(args, 'max_entropy_inference', False)}")
            else:
                print(f"   Entropy coefficient: 0.01 (standard)")
                print(f"   Inference mode: Deterministic")
        elif getattr(args, 'use_discrete_ppo', False):
            print(f"   Discrete PPO epochs: {getattr(args, 'ppo_epochs', 10)}")
            print(f"   Discrete PPO batch_size: {getattr(args, 'ppo_batch_size', 64)}")
            print(f"   Discrete PPO clip_range: {getattr(args, 'ppo_clip_range', 0.2)}")
            print(f"   Discrete PPO gamma: {getattr(args, 'ppo_gamma', 0.99)}")
            print(f"   Discrete PPO GAE lambda: {getattr(args, 'ppo_gae_lambda', 0.95)}")
        elif getattr(args, 'use_ppo', False):
            print(f"   PPO n_steps: {getattr(args, 'ppo_n_steps', 2048)}")
            print(f"   PPO batch_size: {getattr(args, 'ppo_batch_size', 64)}")
            print(f"   PPO n_epochs: {getattr(args, 'ppo_epochs', 10)}")
            print(f"   PPO gamma: {getattr(args, 'ppo_gamma', 0.99)}")
            print(f"   PPO clip_range: {getattr(args, 'ppo_clip_range', 0.2)}")
        else:
            print(f"   Exploration: {args.online_epsilon_start} → {args.online_epsilon_end}")
            print(f"   Memory size: {args.online_memory_size}")
        
        print(f"   Policy will be saved to: {args.online_policy_save_path}")
        print(f"   Resume capability: {not args.no_resume} (checkpoints every {args.checkpoint_freq} steps)")
        print(f"   Checkpoint directory: {args.checkpoint_dir}")
        print(f"   Training seed: {args.training_seed}")
        
        if getattr(args, 'use_ppo', False) or getattr(args, 'continuous_action_space', False):
            print(f"   Action Space: Continuous (total_tokens: 16-128, depth: 2-8, top_k: 2-32)")
        else:
            print(f"   Action Space: Discrete (valid combinations from 6×6×5=180 total bins)")
        if not args.online_inference_only:
            print(f"   Training: Questions repeated {args.online_repeat_factor}x and shuffled")
        else:
            print(f"   Inference: Questions processed in original order")
        if args.online_policy_path:
            print(f"   Loading existing policy from: {args.online_policy_path}")
    elif args.use_rl_policy:
        print("\n🤖 Offline RL Policy Mode: Tree parameters will be predicted dynamically by RL policy")
        print(f"   Fallback parameters: total_token={args.total_token}, depth={args.depth}, top_k={args.top_k}")
        print(f"   RL policy path: {args.rl_policy_path}")
        if getattr(args, 'use_stepwise_rl', False):
            print("⚠️  Warning: Step-wise RL ignored in offline RL mode (requires online learning)")
    else:
        print(f"\n⚙️  Fixed Parameter Mode: total_token={args.total_token}, depth={args.depth}, top_k={args.top_k}")
        if args.collect_rl_data:
            print("   Data collection enabled for future RL training")
        if getattr(args, 'use_stepwise_rl', False):
            print("⚠️  Warning: Step-wise RL ignored in fixed parameter mode (requires online RL)")

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = args.question_file if args.question_file else f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )

    reorg_answer_file(answer_file)
