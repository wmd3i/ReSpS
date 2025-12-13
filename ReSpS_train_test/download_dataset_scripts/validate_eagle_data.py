#!/usr/bin/env python3
"""
Validate and analyze EAGLE training data format.
Check data quality and provide statistics.

Usage:
python validate_eagle_data.py --input eagle_sharegpt.jsonl
"""

import json
import argparse
from collections import Counter
import statistics


def validate_eagle_data(input_file, sample_size=1000):
    """
    Validate EAGLE training data format and provide statistics.
    """
    print(f"Validating EAGLE data: {input_file}")
    
    total_conversations = 0
    valid_conversations = 0
    conversation_lengths = []
    turn_lengths = []
    role_patterns = Counter()
    
    errors = []
    sample_conversations = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                total_conversations += 1
                
                # Check required fields
                if 'id' not in data or 'conversations' not in data:
                    errors.append(f"Line {line_num}: Missing 'id' or 'conversations' field")
                    continue
                
                conversations = data['conversations']
                if not isinstance(conversations, list) or len(conversations) < 2:
                    errors.append(f"Line {line_num}: Invalid conversations structure")
                    continue
                
                # Validate conversation structure
                is_valid = True
                pattern = []
                
                for i, turn in enumerate(conversations):
                    if not isinstance(turn, dict) or 'from' not in turn or 'value' not in turn:
                        errors.append(f"Line {line_num}: Invalid turn structure at index {i}")
                        is_valid = False
                        break
                    
                    role = turn['from']
                    content = turn['value']
                    pattern.append(role)
                    
                    if not content.strip():
                        errors.append(f"Line {line_num}: Empty content at index {i}")
                        is_valid = False
                        break
                    
                    turn_lengths.append(len(content.split()))
                
                if is_valid:
                    valid_conversations += 1
                    conversation_lengths.append(len(conversations))
                    role_patterns[tuple(pattern)] += 1
                    
                    # Collect samples
                    if len(sample_conversations) < sample_size and total_conversations % max(1, total_conversations // sample_size) == 0:
                        sample_conversations.append(data)
            
            except json.JSONDecodeError:
                errors.append(f"Line {line_num}: Invalid JSON")
            except Exception as e:
                errors.append(f"Line {line_num}: {str(e)}")
    
    # Print statistics
    print(f"\n=== Validation Results ===")
    print(f"Total conversations: {total_conversations}")
    print(f"Valid conversations: {valid_conversations}")
    print(f"Success rate: {valid_conversations/total_conversations*100:.2f}%" if total_conversations > 0 else "N/A")
    
    if conversation_lengths:
        print(f"\n=== Conversation Statistics ===")
        print(f"Average conversation length: {statistics.mean(conversation_lengths):.2f} turns")
        print(f"Median conversation length: {statistics.median(conversation_lengths):.2f} turns")
        print(f"Min/Max conversation length: {min(conversation_lengths)}/{max(conversation_lengths)} turns")
    
    if turn_lengths:
        print(f"\n=== Turn Statistics ===")
        print(f"Average turn length: {statistics.mean(turn_lengths):.2f} words")
        print(f"Median turn length: {statistics.median(turn_lengths):.2f} words")
        print(f"Min/Max turn length: {min(turn_lengths)}/{max(turn_lengths)} words")
    
    print(f"\n=== Most Common Role Patterns ===")
    for pattern, count in role_patterns.most_common(10):
        pattern_str = " → ".join(pattern)
        print(f"{pattern_str}: {count} ({count/valid_conversations*100:.1f}%)")
    
    if errors:
        print(f"\n=== Errors (showing first 20) ===")
        for error in errors[:20]:
            print(error)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors")
    
    # Show samples
    if sample_conversations:
        print(f"\n=== Sample Conversations (first 3) ===")
        for i, conv in enumerate(sample_conversations[:3]):
            print(f"\nSample {i+1} (ID: {conv['id']}):")
            for j, turn in enumerate(conv['conversations'][:4]):  # Show first 4 turns
                print(f"  {turn['from']}: {turn['value'][:100]}{'...' if len(turn['value']) > 100 else ''}")
            if len(conv['conversations']) > 4:
                print(f"  ... and {len(conv['conversations']) - 4} more turns")
    
    return {
        'total_conversations': total_conversations,
        'valid_conversations': valid_conversations,
        'success_rate': valid_conversations/total_conversations if total_conversations > 0 else 0,
        'errors': errors
    }


def main():
    parser = argparse.ArgumentParser(description='Validate EAGLE training data')
    parser.add_argument('--input', required=True, help='Input EAGLE JSONL file')
    parser.add_argument('--sample-size', type=int, default=1000, help='Number of sample conversations to collect')
    
    args = parser.parse_args()
    
    validate_eagle_data(args.input, args.sample_size)


if __name__ == "__main__":
    main()
