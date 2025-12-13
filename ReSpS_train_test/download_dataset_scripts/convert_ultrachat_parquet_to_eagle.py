#!/usr/bin/env python3
"""
Convert UltraChat-200K Parquet files to EAGLE training format.
UltraChat Parquet format -> EAGLE conversation format

Usage:
python convert_ultrachat_parquet_to_eagle.py --input-dir /path/to/ultrachat/data --output eagle_ultrachat.jsonl
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob


def convert_ultrachat_parquet_to_eagle(input_dir, output_file, max_conversations=None, file_pattern="train_sft*.parquet"):
    """
    Convert UltraChat Parquet files to EAGLE format.
    
    UltraChat format in Parquet:
    {
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
    print(f"Converting UltraChat Parquet data from {input_dir} to {output_file}")
    
    # Find all matching parquet files
    input_path = Path(input_dir)
    parquet_files = list(input_path.glob(file_pattern))
    
    if not parquet_files:
        print(f"No Parquet files found matching pattern '{file_pattern}' in {input_dir}")
        print("Available files:")
        for file in input_path.glob("*.parquet"):
            print(f"  {file.name}")
        return
    
    print(f"Found {len(parquet_files)} Parquet files:")
    for f in parquet_files:
        print(f"  {f.name}")
    
    converted_count = 0
    skipped_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for parquet_file in parquet_files:
            print(f"\nProcessing {parquet_file.name}...")
            
            try:
                # Read parquet file
                df = pd.read_parquet(parquet_file)
                print(f"  Loaded {len(df)} rows")
                print(f"  Columns: {list(df.columns)}")
                
                # Show first few rows structure for debugging
                if len(df) > 0:
                    print(f"  Sample row type: {type(df.iloc[0])}")
                    if 'messages' in df.columns:
                        sample_messages = df.iloc[0]['messages']
                        print(f"  Sample messages type: {type(sample_messages)}")
                        if hasattr(sample_messages, 'tolist'):
                            sample_messages = sample_messages.tolist()
                        if len(sample_messages) > 0:
                            print(f"  First message type: {type(sample_messages[0])}")
                            print(f"  First message sample: {sample_messages[0] if len(str(sample_messages[0])) < 200 else str(sample_messages[0])[:200] + '...'}")
                
                # Process each row
                for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {parquet_file.name}"):
                    if max_conversations and converted_count >= max_conversations:
                        break
                    
                    try:
                        # Extract messages - handle both dict-like and pandas Series
                        if hasattr(row, 'messages'):
                            messages = row.messages
                        else:
                            messages = row.get('messages', [])
                        
                        # Convert numpy array or pandas series to list if needed
                        if hasattr(messages, 'tolist'):
                            messages = messages.tolist()
                        elif not isinstance(messages, (list, tuple)):
                            messages = []
                        
                        # Validate messages
                        if len(messages) < 2:
                            skipped_count += 1
                            continue
                        
                        # Convert messages to EAGLE conversation format
                        conversations = []
                        valid_conversation = True
                        
                        for msg in messages:
                            # Handle different message formats
                            if isinstance(msg, dict):
                                role = msg.get('role', '')
                                content = msg.get('content', '')
                            else:
                                # Skip invalid message formats
                                continue
                            
                            if not content or not content.strip():
                                valid_conversation = False
                                break
                            
                            if role == 'user':
                                conversations.append({
                                    "from": "human",
                                    "value": content.strip()
                                })
                            elif role == 'assistant':
                                conversations.append({
                                    "from": "gpt", 
                                    "value": content.strip()
                                })
                            elif role == 'system':
                                # Skip system messages for now, but don't invalidate
                                continue
                            else:
                                valid_conversation = False
                                break
                        
                        if not valid_conversation or len(conversations) < 2:
                            skipped_count += 1
                            continue
                        
                        # Ensure alternating pattern starting with human
                        if conversations[0]["from"] != "human":
                            skipped_count += 1
                            continue
                        
                        # Create EAGLE format entry
                        eagle_entry = {
                            "id": f"ultrachat_{converted_count}",
                            "conversations": conversations
                        }
                        
                        f_out.write(json.dumps(eagle_entry, ensure_ascii=False) + '\n')
                        converted_count += 1
                        
                    except Exception as e:
                        print(f"  Error processing row {idx}: {e}")
                        skipped_count += 1
                        continue
                
                if max_conversations and converted_count >= max_conversations:
                    print(f"  Reached max conversations limit ({max_conversations})")
                    break
                    
            except Exception as e:
                print(f"Error processing {parquet_file}: {e}")
                continue
    
    print(f"\nConversion complete!")
    print(f"Converted: {converted_count} conversations")
    print(f"Skipped: {skipped_count} conversations")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert UltraChat Parquet to EAGLE format')
    parser.add_argument('--input-dir', required=True, help='Input directory containing Parquet files')
    parser.add_argument('--output', required=True, help='Output EAGLE JSONL file')
    parser.add_argument('--max-conversations', type=int, help='Maximum number of conversations to convert')
    parser.add_argument('--file-pattern', default='train_sft*.parquet', 
                       help='Pattern to match Parquet files (default: train_sft*.parquet)')
    
    args = parser.parse_args()
    
    convert_ultrachat_parquet_to_eagle(args.input_dir, args.output, args.max_conversations, args.file_pattern)


if __name__ == "__main__":
    main()
