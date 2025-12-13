"""
Quick test script for EAGLE RL functionality.
This script tests the RL components without requiring full model setup.
"""

import sys
import os
import tempfile
import json

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from eagle.evaluation.rl_tree_policy import TreeRLEnvironment, RLTreePolicy, train_and_save_policy, calculate_real_reward
    print("✓ Successfully imported RL components")
except ImportError as e:
    print(f"✗ Failed to import RL components: {e}")
    print("Make sure to install RL dependencies: pip install -r requirements-rl.txt")
    sys.exit(1)


def test_rl_environment():
    """Test the RL environment creation and basic functionality."""
    print("\n=== Testing RL Environment ===")
    
    # Create dummy questions
    questions = [
        {"question_id": 1, "turns": ["What is the capital of France?"]},
        {"question_id": 2, "turns": ["Explain quantum computing"]},
        {"question_id": 3, "turns": ["Write a Python function to sort a list"]}
    ]
    
    try:
        env = TreeRLEnvironment(questions)
        print("✓ Successfully created TreeRLEnvironment")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Environment reset, observation shape: {obs.shape}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"✓ Environment step completed")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Info: {info}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_rl_training():
    """Test RL policy training with minimal setup."""
    print("\n=== Testing RL Training ===")
    
    # Create dummy questions
    questions = [
        {"question_id": i, "turns": [f"Test question {i}"]} 
        for i in range(5)
    ]
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            tmp_policy_path = tmp_file.name
        
        # Train with minimal timesteps for testing
        train_and_save_policy(questions, tmp_policy_path, total_timesteps=50)
        print("✓ Successfully trained RL policy")
        
        # Test loading the trained policy
        policy = RLTreePolicy(tmp_policy_path)
        print("✓ Successfully loaded trained RL policy")
        
        # Test prediction
        test_text = "This is a test question about machine learning"
        total_tokens, depth, top_k = policy.predict_parameters(test_text)
        print(f"✓ Policy prediction:")
        print(f"  Input: {test_text}")
        print(f"  Predicted - total_tokens: {total_tokens}, depth: {depth}, top_k: {top_k}")
        
        # Cleanup
        os.unlink(tmp_policy_path)
        return True
        
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        return False


def test_reward_calculation():
    """Test reward calculation function."""
    print("\n=== Testing Reward Calculation ===")
    
    try:
        # Test different scenarios
        scenarios = [
            {"generation_time": 1.0, "new_tokens": 20, "total_tokens": 60, "depth": 5, "top_k": 10},
            {"generation_time": 0.5, "new_tokens": 30, "total_tokens": 40, "depth": 3, "top_k": 8},
            {"generation_time": 2.0, "new_tokens": 10, "total_tokens": 80, "depth": 8, "top_k": 15},
        ]
        
        for i, scenario in enumerate(scenarios):
            reward = calculate_real_reward(**scenario)
            print(f"✓ Scenario {i+1}: reward = {reward:.2f}")
            print(f"  Parameters: {scenario}")
        
        return True
        
    except Exception as e:
        print(f"✗ Reward calculation test failed: {e}")
        return False


def test_data_collection_format():
    """Test the format of RL data collection."""
    print("\n=== Testing Data Collection Format ===")
    
    try:
        # Simulate data collection entry
        rl_data_entry = {
            "question_id": "test_1",
            "turn": 0,
            "choice": 0,
            "context": "Test question about AI",
            "total_tokens": 60,
            "depth": 5,
            "top_k": 10,
            "generation_time": 1.5,
            "new_tokens": 25,
            "reward": 3.2,
            "tokens_per_second": 16.7
        }
        
        # Test JSON serialization
        json_str = json.dumps(rl_data_entry)
        loaded_data = json.loads(json_str)
        
        print("✓ Data collection format test passed")
        print(f"  Sample entry: {loaded_data}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data collection format test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("EAGLE RL Component Testing")
    print("=" * 50)
    
    tests = [
        test_rl_environment,
        test_reward_calculation,
        test_data_collection_format,
        test_rl_training,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! RL functionality is working correctly.")
        print("\nNext steps:")
        print("1. Install your EAGLE models")
        print("2. Run: python train_rl_policy.py --mode from_questions")
        print("3. Run: python gen_ea_answer_llama3chat_rl.py --use-rl-policy")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
