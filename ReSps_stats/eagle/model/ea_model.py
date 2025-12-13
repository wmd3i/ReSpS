import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

# Import profiler for overhead measurement
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

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,  # Increased from 2048 to accommodate larger sequences
            log=False,
            is_llama3=False,
            total_tokens=None,
            depth=None,
            tree_top_k=None,
            # New step-wise RL parameters
            rl_policy=None,
            training_mode=False,
            step_rewards_callback=None,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        # Store original parameters for restoration later
        original_total_tokens = self.ea_layer.total_tokens
        original_depth = self.ea_layer.depth
        original_top_k = self.ea_layer.top_k
        
        # For step-wise RL: track step-level metrics
        step_rewards = []
        step_actions = []
        step_states = []
        use_stepwise_rl = rl_policy is not None
        
        # Track unique actions and their first appearance step
        unique_actions = {}  # action_tuple -> first_step_number
        generation_speeds = []  # Store generation speed for each step
        
        # Temporarily update parameters if provided (fallback values for non-RL mode)
        if total_tokens is not None:
            self.ea_layer.total_tokens = total_tokens - 1  # Adjust as in line 163
        if depth is not None:
            self.ea_layer.depth = depth
        if tree_top_k is not None:
            self.ea_layer.top_k = tree_top_k


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        
        # For step-wise RL: get initial state for first prediction
        if use_stepwise_rl:
            # Check if this is an optimized policy that supports EAGLE-3 features
            is_optimized_policy = hasattr(rl_policy, 'use_eagle3_features') and rl_policy.use_eagle3_features
            
            if training_mode:
                # Encode current context for RL policy
                current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                initial_state = current_text
                
                # For optimized policies, we don't have hidden states yet, so pass None initially
                if is_optimized_policy:
                    # step_total_tokens, step_depth, step_top_k = rl_policy.predict_parameters(
                        # context=initial_state, hidden_states=None, training_mode=training_mode
                    # )
                    step_total_tokens, step_depth, step_top_k = 96, 8, 20
                else:
                    # Traditional policy interface
                    step_total_tokens, step_depth, step_top_k = rl_policy.predict_parameters(
                        initial_state, training_mode=training_mode
                    )
            else:
                with torch.no_grad():
                    # Encode current context for RL policy
                    current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    initial_state = current_text
                    
                    # For optimized policies, we don't have hidden states yet, so pass None initially
                    if is_optimized_policy:
                        # step_total_tokens, step_depth, step_top_k = rl_policy.predict_parameters(
                            # context=initial_state, hidden_states=None, training_mode=training_mode
                        # )
                        step_total_tokens, step_depth, step_top_k = 96, 8, 20
                    else:
                        # Traditional policy interface
                        step_total_tokens, step_depth, step_top_k = rl_policy.predict_parameters(
                            initial_state, training_mode=training_mode
                        )
            
            # Apply bounds checking for initial parameters
            # if step_total_tokens:
            #     # Ensure total_tokens doesn't exceed available buffer space
            #     max_available_tokens = max_length - input_ids.shape[1] - 10  # Leave some buffer
                
            #     # Ensure max_available_tokens is at least 1
            #     if max_available_tokens < 1:
            #         max_available_tokens = 1
                
            #     if step_total_tokens > max_available_tokens:
            #         step_total_tokens = max_available_tokens
                
            #     # Ensure total_tokens is at least 1 to prevent negative values
            #     if step_total_tokens < 1:
            #         step_total_tokens = 1
            
            # Store step info for training
            if True: # training_mode:
                step_states.append(initial_state)
                step_actions.append((step_total_tokens, step_depth, step_top_k))
                
                # Track unique actions
                action_tuple = (step_total_tokens, step_depth, step_top_k)
                if action_tuple not in unique_actions:
                    unique_actions[action_tuple] = 0  # Step 0 for initial action
            
            # print(f"Step 0 RL params: tt={step_total_tokens}, d={step_depth}, k={step_top_k}")
        else:
            # Use provided or default parameters
            step_total_tokens = total_tokens
            step_depth = depth
            step_top_k = tree_top_k
        
        # prefill - model inference, disable gradients for efficiency
        with torch.no_grad():
        #     # Final safety check before calling initialize_tree
        #     if step_total_tokens:
        #         max_available_tokens = max_length - input_ids.shape[1] - 10  # Leave some buffer
                
        #         # Ensure max_available_tokens is at least 1
        #         if max_available_tokens < 1:
        #             print(f"Warning: max_available_tokens ({max_available_tokens}) too small, setting to minimum value 1")
        #             max_available_tokens = 1
                
        #         if step_total_tokens > max_available_tokens:
        #             print(f"Warning: RL predicted total_tokens ({step_total_tokens}) too large, clamping to {max_available_tokens}")
        #             step_total_tokens = max_available_tokens
                
        #         # Ensure total_tokens is at least 1 to prevent negative values
        #         if step_total_tokens < 1:
        #             print(f"Warning: Clamped total_tokens ({step_total_tokens}) too small, setting to minimum value 1")
        #             step_total_tokens = 1
            
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
                input_ids, self, past_key_values, logits_processor, step_total_tokens, step_depth, step_top_k
            )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        
        # Ensure max_length doesn't become too small
        min_safe_length = input_ids.shape[1] + 20  # At least 20 tokens buffer
        if max_length < min_safe_length:
            print(f"Warning: max_length ({max_length}) too small, adjusting to minimum safe length ({min_safe_length})")
            max_length = min_safe_length
        
        for idx in range(max_length):
            # For step-wise RL: predict parameters at each step (except first which was done above)
            step_start_time = time.time()
            
            if use_stepwise_rl and idx > 0:  # Skip first iteration since we already predicted
                # Check if this is an optimized policy that supports EAGLE-3 features
                is_optimized_policy = hasattr(rl_policy, 'use_eagle3_features') and rl_policy.use_eagle3_features
                
                if training_mode:
                    # Get current context for RL policy
                    current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    current_state = current_text
                    
                    # For optimized policies, pass EAGLE-3 hidden states if available - WITH OVERHEAD MEASUREMENT
                    if is_optimized_policy and 'hidden_state' in locals():
                        with overhead_profiler.time_component("RL_Policy_Stepwise_Optimized_Training"):
                            step_total_tokens, step_depth, step_top_k = rl_policy.predict_parameters(
                                context=current_state, hidden_states=hidden_state, training_mode=training_mode
                            )
                    else:
                        # Traditional policy interface - WITH OVERHEAD MEASUREMENT
                        with overhead_profiler.time_component("RL_Policy_Stepwise_Traditional_Training"):
                            step_total_tokens, step_depth, step_top_k = rl_policy.predict_parameters(
                                current_state, training_mode=training_mode
                            )
                    
                    # Store step info for training
                    if True: # training_mode:
                        step_states.append(current_state)
                        step_actions.append((step_total_tokens, step_depth, step_top_k))
                        
                        # Track unique actions
                        action_tuple = (step_total_tokens, step_depth, step_top_k)
                        if action_tuple not in unique_actions:
                            unique_actions[action_tuple] = idx
                    # if len(step_rewards) % 30 == 0:
                    #     print(f"  Step {idx} RL params: tt={step_total_tokens}, d={step_depth}, k={step_top_k}")
                else:
                    with torch.no_grad():
                        # Get current context for RL policy
                        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                        current_state = current_text
                        
                        # For optimized policies, pass EAGLE-3 hidden states if available - WITH OVERHEAD MEASUREMENT
                        if is_optimized_policy and 'hidden_state' in locals():
                            with overhead_profiler.time_component("RL_Policy_Stepwise_Optimized_Inference"):
                                step_total_tokens, step_depth, step_top_k = rl_policy.predict_parameters(
                                    context=current_state, hidden_states=hidden_state, training_mode=training_mode
                                )
                        else:
                            # Traditional policy interface - WITH OVERHEAD MEASUREMENT
                            with overhead_profiler.time_component("RL_Policy_Stepwise_Traditional_Inference"):
                                step_total_tokens, step_depth, step_top_k = rl_policy.predict_parameters(
                                    current_state, training_mode=training_mode
                                )
                        
                        # Store step info for training
                        if True: # training_mode:
                            step_states.append(current_state)
                            step_actions.append((step_total_tokens, step_depth, step_top_k))
                            
                            # Track unique actions
                            action_tuple = (step_total_tokens, step_depth, step_top_k)
                            if action_tuple not in unique_actions:
                                unique_actions[action_tuple] = idx
                        # if len(step_rewards) % 30 == 0:
                        #     print(f"  Step {idx} RL params: tt={step_total_tokens}, d={step_depth}, k={step_top_k}")
                        # print(f"Step {idx} RL params: tt={step_total_tokens}, d={step_depth}, k={step_top_k}")
                
                # Update model parameters for this step with bounds checking
                # if step_total_tokens:
                #     # Ensure total_tokens doesn't exceed available buffer space
                #     max_available_tokens = max_length - input_ids.shape[1] - 10  # Leave some buffer
                    
                #     # Ensure max_available_tokens is at least 1
                #     if max_available_tokens < 1:
                #         print(f"Warning: Step {idx} max_available_tokens ({max_available_tokens}) too small, setting to minimum value 1")
                #         max_available_tokens = 1
                    
                #     if step_total_tokens > max_available_tokens:
                #         print(f"Warning: Step {idx} RL predicted total_tokens ({step_total_tokens}) too large, clamping to {max_available_tokens}")
                #         step_total_tokens = max_available_tokens
                    
                #     # Ensure total_tokens is at least 1 to prevent negative values
                #     if step_total_tokens < 1:
                #         print(f"Warning: Step {idx} clamped total_tokens ({step_total_tokens}) too small, setting to minimum value 1")
                #         step_total_tokens = 1
                if step_total_tokens:
                    self.ea_layer.total_tokens = step_total_tokens - 1
                if step_depth:
                    self.ea_layer.depth = step_depth
                if step_top_k:
                    self.ea_layer.top_k = step_top_k
            
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            # Model inference parts: disable gradients for efficiency
            with torch.no_grad():
                draft_tokens = draft_tokens.to(input_ids.device)
                # Target model forward, get logits - WITH OVERHEAD MEASUREMENT
                with overhead_profiler.time_component("Target_Model_Forward"):
                    logits, hidden_state_new, outputs = tree_decoding(
                        self,
                        draft_tokens,
                        past_key_values,
                        tree_position_ids,
                        input_ids,
                        retrieve_indices,
                    )
                # retrieve_indices=tree_buffers["retrieve_indices"]
                # logits = logits[0, retrieve_indices]
                draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                candidates = draft_tokens[0, retrieve_indices]
                # verification
                best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, logits_processor
                )
                # print(accept_length)
            
            # Calculate step reward for RL training
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_tokens_generated = accept_length + 1  # Number of tokens accepted in this step
            
            if use_stepwise_rl and step_time > 0 and step_tokens_generated > 0:
                # Calculate reward: tokens per second for this step
                step_reward = step_tokens_generated / step_time
                step_rewards.append(step_reward)
                generation_speeds.append(step_tokens_generated / step_time)
                
                # Update policy if in training mode
                if training_mode and len(step_rewards) >= 1:
                    try:
                        # Pass generation_time and new_tokens for optimized policies
                        rl_policy.update_policy(step_reward, step_time, step_tokens_generated, training_mode)
                        # if len(step_rewards) % 30 == 0:  # Log every 30 steps
                        #     print(f"  Step {idx} reward: {step_reward:.2f} tok/s (accepted: {step_tokens_generated})")
                    except Exception as e:
                        
                        raise e
                        print(f"  Warning: RL policy update failed at step {idx}: {e}")
            
            # Adjusting the input sequence, draft model forward - WITH OVERHEAD MEASUREMENT  
            with torch.no_grad():
                with overhead_profiler.time_component("Draft_Model_Forward"):
                    input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                        input_ids,
                        candidates,
                        best_candidate,
                        accept_length,
                        retrieve_indices,
                        logits_processor,
                        new_token,
                        past_key_values_data,
                        current_length_data,
                        self,
                        hidden_state_new,
                        sample_p,
                        return_hidden_states=use_stepwise_rl  # Return hidden states for RL policies
                    )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        
        # Reset policy state after generation is complete
        if use_stepwise_rl and rl_policy is not None:
            if hasattr(rl_policy, 'reset'):
                rl_policy.reset(training_mode)
        
        # Print summary statistics if using step-wise RL
        if use_stepwise_rl: # and training_mode:
            # print("\n=== Step-wise RL Summary ===")
            
            # Print unique actions and their first appearance
            # print(f"Unique actions found: {len(unique_actions)}")
            for action, first_step in sorted(unique_actions.items(), key=lambda x: x[1]):
                tokens, depth, top_k = action
                print(f"  Action (tokens={tokens}, depth={depth}, top_k={top_k}) first appeared at step {first_step}")
            
            # Calculate and print mean statistics
            if step_rewards:
                mean_reward = sum(step_rewards) / len(step_rewards)
                print(f"Mean reward: {mean_reward:.2f} tok/s over {len(step_rewards)} steps")
            
            if generation_speeds:
                mean_speed = sum(generation_speeds) / len(generation_speeds)
                print(f"Mean generation speed: {mean_speed:.2f} tok/s over {len(generation_speeds)} steps")
            
            # print("=== End Summary ===\n")
        
        # Restore original parameters
        self.ea_layer.total_tokens = original_total_tokens
        self.ea_layer.depth = original_depth
        self.ea_layer.top_k = original_top_k
        
        # Return step-wise RL information if used
        if use_stepwise_rl and step_rewards_callback and training_mode:
            step_rewards_callback({
                'step_rewards': step_rewards,
                'step_actions': step_actions, 
                'step_states': step_states,
                'total_steps': len(step_rewards)
            })
        
        if not log:
            if use_stepwise_rl:
                return input_ids, step_rewards, len(step_rewards)
            else:
                return input_ids
        else:
            if use_stepwise_rl:
                return input_ids, new_token, idx, step_rewards, len(step_rewards)
            else:
                return input_ids, new_token, idx

    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,  # Increased from 2048 to accommodate larger sequences
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,  # Increased from 2048 to accommodate larger sequences
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,  # Increased from 2048 to accommodate larger sequences
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
