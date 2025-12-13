"""
Training script for RL policy that optimizes EAGLE tree generation parameters.
This script trains a PPO policy using collected generation data.
"""

import argparse
import json
import os
from fastchat.llm_judge.common import load_questions
from eagle.evaluation.rl_tree_policy import TreeRLEnvironment, train_and_save_policy


def load_rl_training_data(data_file):
    """
    Load RL training data from JSONL file.
    
    Args:
        data_file: Path to JSONL file containing RL training data
    
    Returns:
        List of training data entries
    """
    data = []
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    return data


def simulate_environment_from_data(data):
    """
    Create simulated questions for training based on collected data.
    This extracts unique contexts to create a training environment.
    
    Args:
        data: List of RL training data entries
    
    Returns:
        List of simulated questions for training
    """
    # Extract unique contexts and create pseudo-questions
    contexts = set()
    for entry in data:
        contexts.add(entry['context'])
    
    questions = []
    for i, context in enumerate(contexts):
        questions.append({
            "question_id": f"rl_train_{i}",
            "turns": [context]
        })
    
    return questions


def train_rl_policy_from_data(data_file, policy_path, total_timesteps=2000):
    """
    Train RL policy using collected data.
    
    Args:
        data_file: Path to collected RL training data
        policy_path: Path to save trained policy
        total_timesteps: Number of training timesteps
    """
    # Load training data
    data = load_rl_training_data(data_file)
    print(f"Loaded {len(data)} training data entries")
    
    if len(data) == 0:
        print("No training data found. Please run evaluation with --collect-rl-data first.")
        return
    
    # Create simulated questions from data
    questions = simulate_environment_from_data(data)
    print(f"Created {len(questions)} training questions")
    
    # Train policy
    train_and_save_policy(questions, policy_path, total_timesteps)
    print(f"Training completed. Policy saved to {policy_path}")


def train_rl_policy_from_questions(question_file, policy_path, total_timesteps=1000):
    """
    Train RL policy using questions directly (without real data).
    This is useful for initial training before collecting real performance data.
    
    Args:
        question_file: Path to question file
        policy_path: Path to save trained policy
        total_timesteps: Number of training timesteps
    """
    questions = load_questions(question_file, None, None)
    print(f"Loaded {len(questions)} questions for training")
    
    train_and_save_policy(questions, policy_path, total_timesteps)
    print(f"Training completed. Policy saved to {policy_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL policy for EAGLE tree optimization")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["from_data", "from_questions"],
        default="from_questions",
        help="Training mode: use collected data or direct questions"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="rl_training_data.jsonl",
        help="Path to collected RL training data (for from_data mode)"
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default="../data/mt_bench/question.jsonl",
        help="Path to question file (for from_questions mode)"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="ppo_tree_policy_discrete.zip",
        help="Path to save trained policy"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000,
        help="Number of training timesteps"
    )
    
    args = parser.parse_args()
    
    print(f"Training RL policy in {args.mode} mode")
    
    if args.mode == "from_data":
        train_rl_policy_from_data(args.data_file, args.policy_path, args.total_timesteps)
    else:
        train_rl_policy_from_questions(args.question_file, args.policy_path, args.total_timesteps)
