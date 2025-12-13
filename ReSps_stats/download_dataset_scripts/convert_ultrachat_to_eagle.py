#!/usr/bin/env python3
"""
Convert UltraChat-200K dataset to EAGLE training format.
UltraChat format -> EAGLE conversation format

Usage:
python convert_ultrachat_to_eagle.py --input UltraChat_200k.jsonl --output eagle_ultrachat.jsonl
"""

import json
import argparse
from tqdm import tqdm
import uuid


def convert_ultrachat_to_eagle(input_file, output_file, max_conversations=None):
    """
    Convert UltraChat format to EAGLE format.
    
    UltraChat format:
    {
        "id": "...",
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    }
    
    EAGLE format:
    {
        "id": "...",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
            ...
        ]
    }
    """
    print(f"Converting UltraChat data from {input_file} to {output_file}")
    
    converted_count = 0
    skipped_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in, desc="Converting conversations"):
                if max_conversations and converted_count >= max_conversations:
                    break
                
                try:
                    item = json.loads(line.strip())
                    
                    # Handle different possible formats
                    messages = item.get('messages', item.get('data', []))
                    if not messages or len(messages) < 2:
                        skipped_count += 1
                        continue
                    
                    # Convert messages to EAGLE conversation format
                    conversations = []
                    valid_conversation = True
                    
                    for i, msg in enumerate(messages):
                        # Handle different field names
                        role = msg.get('role', msg.get('from', ''))
                        content = msg.get('content', msg.get('value', ''))
                        
                        if not content.strip():
                            valid_conversation = False
                            break
                        
                        if role in ['user', 'human']:
                            conversations.append({
                                "from": "human",
                                "value": content.strip()
                            })
                        elif role in ['assistant', 'gpt']:
                            conversations.append({
                                "from": "gpt",
                                "value": content.strip()
                            })
                        else:
                            # Skip system messages or unknown roles for now
                            if role != 'system':
                                valid_conversation = False
                                break
                    
                    if not valid_conversation or len(conversations) < 2:
                        skipped_count += 1
                        continue
                    
                    # Ensure alternating pattern (start with human)
                    if conversations[0]["from"] != "human":
                        skipped_count += 1
                        continue
                    
                    # Create EAGLE format entry
                    eagle_entry = {
                        "id": item.get('id', f"ultrachat_{converted_count}"),
                        "conversations": conversations
                    }
                    
                    f_out.write(json.dumps(eagle_entry, ensure_ascii=False) + '\n')
                    converted_count += 1
                    
                except Exception as e:
                    print(f"Error processing line {converted_count + skipped_count}: {e}")
                    skipped_count += 1
                    continue
    
    print(f"Conversion complete!")
    print(f"Converted: {converted_count} conversations")
    print(f"Skipped: {skipped_count} conversations")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert UltraChat to EAGLE format')
    parser.add_argument('--input', required=True, help='Input UltraChat JSONL file')
    parser.add_argument('--output', required=True, help='Output EAGLE JSONL file')
    parser.add_argument('--max-conversations', type=int, help='Maximum number of conversations to convert')
    
    args = parser.parse_args()
    
    convert_ultrachat_to_eagle(args.input, args.output, args.max_conversations)


if __name__ == "__main__":
    main()
