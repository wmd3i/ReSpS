#!/usr/bin/env python3
"""
Convert ShareGPT dataset to EAGLE training format.
ShareGPT format -> EAGLE conversation format

Usage:
python convert_sharegpt_to_eagle.py --input ShareGPT_V3_unfiltered_cleaned_split.json --output eagle_sharegpt.jsonl
"""

import json
import argparse
from tqdm import tqdm
import uuid


def convert_sharegpt_to_eagle(input_file, output_file, max_conversations=None):
    """
    Convert ShareGPT format to EAGLE format.
    
    ShareGPT format:
    {
        "id": "...",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
            ...
        ]
    }
    
    EAGLE format (same structure, just ensuring consistency):
    {
        "id": "...",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
            ...
        ]
    }
    """
    print(f"Converting ShareGPT data from {input_file} to {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_count = 0
    skipped_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="Converting conversations"):
            if max_conversations and converted_count >= max_conversations:
                break
                
            try:
                # Validate conversation structure
                conversations = item.get('conversations', [])
                if not conversations or len(conversations) < 2:
                    skipped_count += 1
                    continue
                
                # Ensure alternating human/gpt pattern
                valid_conversation = True
                cleaned_conversations = []
                
                for i, turn in enumerate(conversations):
                    if i % 2 == 0:  # Even indices should be human
                        if turn['from'] not in ['human', 'user']:
                            valid_conversation = False
                            break
                        cleaned_conversations.append({
                            "from": "human",
                            "value": turn['value'].strip()
                        })
                    else:  # Odd indices should be gpt
                        if turn['from'] not in ['gpt', 'assistant']:
                            valid_conversation = False
                            break
                        cleaned_conversations.append({
                            "from": "gpt", 
                            "value": turn['value'].strip()
                        })
                
                if not valid_conversation or len(cleaned_conversations) < 2:
                    skipped_count += 1
                    continue
                
                # Create EAGLE format entry
                eagle_entry = {
                    "id": item.get('id', f"sharegpt_{converted_count}"),
                    "conversations": cleaned_conversations
                }
                
                f.write(json.dumps(eagle_entry, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except Exception as e:
                print(f"Error processing conversation {item.get('id', 'unknown')}: {e}")
                skipped_count += 1
                continue
    
    print(f"Conversion complete!")
    print(f"Converted: {converted_count} conversations")
    print(f"Skipped: {skipped_count} conversations")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert ShareGPT to EAGLE format')
    parser.add_argument('--input', required=True, help='Input ShareGPT JSON file')
    parser.add_argument('--output', required=True, help='Output EAGLE JSONL file')
    parser.add_argument('--max-conversations', type=int, help='Maximum number of conversations to convert')
    
    args = parser.parse_args()
    
    convert_sharegpt_to_eagle(args.input, args.output, args.max_conversations)


if __name__ == "__main__":
    main()
