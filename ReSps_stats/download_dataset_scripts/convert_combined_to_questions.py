#!/usr/bin/env python3
"""
Convert EAGLE combined dataset to MT-Bench style question format for RL training.

Usage:
    python download_dataset_scripts/convert_combined_to_questions.py \
        --input ./training_data/processed/eagle_combined.jsonl \
        --output ./eagle/data/rl_training/question.jsonl \
        --max-questions 1000
"""

import argparse
import json
import os
from pathlib import Path
import random


def convert_conversation_to_question(conversation_data, question_id):
    """
    Convert EAGLE conversation format to MT-Bench question format.
    
    EAGLE format:
    {
        "id": "QWJhYvA_0", 
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
            ...
        ]
    }
    
    MT-Bench format:
    {
        "question_id": 81,
        "category": "writing", 
        "turns": ["First question", "Follow-up question"]
    }
    """
    conversations = conversation_data.get("conversations", [])
    
    # Extract human turns (questions) from the conversation
    human_turns = []
    for turn in conversations:
        if turn.get("from") == "human":
            human_turns.append(turn.get("value", "").strip())
    
    # Skip if no human turns or empty content
    if not human_turns or not any(turn for turn in human_turns):
        return None
    
    # For RL training, we primarily use the first human turn as the main question
    # If there are multiple human turns, we can use the second as a follow-up
    turns = [human_turns[0]]
    
    # Add a generic follow-up if there's a second human turn, otherwise create one
    if len(human_turns) > 1:
        turns.append(human_turns[1])
    else:
        # Create a generic follow-up question for multi-turn evaluation
        turns.append("Can you elaborate on your previous response and provide more specific details?")
    
    # Determine category based on content heuristics
    first_turn_lower = human_turns[0].lower()
    if any(keyword in first_turn_lower for keyword in ["write", "compose", "draft", "create", "blog", "email", "letter"]):
        category = "writing"
    elif any(keyword in first_turn_lower for keyword in ["code", "program", "function", "algorithm", "debug", "python", "java", "javascript"]):
        category = "coding"
    elif any(keyword in first_turn_lower for keyword in ["math", "calculate", "solve", "equation", "formula", "number"]):
        category = "math"
    elif any(keyword in first_turn_lower for keyword in ["analyze", "explain", "reasoning", "why", "how", "logic"]):
        category = "reasoning"
    elif any(keyword in first_turn_lower for keyword in ["roleplay", "pretend", "imagine", "act as", "you are"]):
        category = "roleplay"
    elif any(keyword in first_turn_lower for keyword in ["extract", "summarize", "information", "data", "facts"]):
        category = "extraction"
    elif any(keyword in first_turn_lower for keyword in ["translate", "language", "meaning", "definition"]):
        category = "humanities"
    else:
        category = "general"
    
    return {
        "question_id": question_id,
        "category": category,
        "turns": turns,
        "original_id": conversation_data.get("id", ""),
        "conversation_length": len(conversations)
    }


def main():
    parser = argparse.ArgumentParser(description="Convert EAGLE combined dataset to MT-Bench question format")
    parser.add_argument("--input", type=str, required=True, help="Input EAGLE combined dataset file")
    parser.add_argument("--output", type=str, required=True, help="Output question file")
    parser.add_argument("--max-questions", type=int, default=None, help="Maximum number of questions to convert")
    parser.add_argument("--min-conversation-length", type=int, default=2, help="Minimum conversation length to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset before conversion")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting EAGLE dataset to MT-Bench format...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max questions: {args.max_questions if args.max_questions else 'All'}")
    print(f"Min conversation length: {args.min_conversation_length}")
    
    # Read input file
    conversations = []
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # Filter by conversation length
                    if len(data.get("conversations", [])) >= args.min_conversation_length:
                        conversations.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue
                    
                # Progress indicator
                if line_num % 50000 == 0:
                    print(f"Processed {line_num} lines, kept {len(conversations)} conversations")
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        return
    
    print(f"Loaded {len(conversations)} conversations")
    
    # Shuffle if requested
    if args.shuffle:
        random.shuffle(conversations)
        print("Dataset shuffled")
    
    # Limit number of conversations if specified
    if args.max_questions and len(conversations) > args.max_questions:
        conversations = conversations[:args.max_questions]
        print(f"Limited to {args.max_questions} conversations")
    
    # Convert conversations to questions
    questions = []
    skipped = 0
    
    for i, conv in enumerate(conversations):
        question = convert_conversation_to_question(conv, i + 1)
        if question:
            questions.append(question)
        else:
            skipped += 1
    
    print(f"Converted {len(questions)} questions, skipped {skipped} invalid conversations")
    
    # Category statistics
    categories = {}
    for q in questions:
        cat = q["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        percentage = (count / len(questions)) * 100
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    # Write output file
    with open(args.output, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Conversion complete!")
    print(f"Output saved to: {args.output}")
    print(f"Total questions: {len(questions)}")
    
    # Show sample questions
    print(f"\nSample questions (first 3):")
    for i, q in enumerate(questions[:3]):
        print(f"\nQuestion {q['question_id']} ({q['category']}):")
        print(f"  Turn 1: {q['turns'][0][:100]}...")
        if len(q['turns']) > 1:
            print(f"  Turn 2: {q['turns'][1][:100]}...")


if __name__ == "__main__":
    main()
