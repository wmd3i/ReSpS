import json
import argparse
from transformers import AutoTokenizer
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Calculate speed ratio between EAGLE and baseline models')
    parser.add_argument('--ea-file', type=str, required=True, 
                        help='Path to EAGLE evaluation JSONL file')
    parser.add_argument('--baseline-file', type=str, required=True, 
                        help='Path to baseline evaluation JSONL file')
    parser.add_argument('--tokenizer-path', type=str, 
                        default="/home/lyh/weights/hf/llama2chat/13B/",
                        help='Path to tokenizer model')
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    jsonl_file = args.ea_file
    jsonl_file_base = args.baseline_file
    
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds=[]
    for datapoint in data:
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens=sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds.append(tokens/times)

    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    total_time=0
    total_token=0
    speeds0=[]
    for datapoint in data:
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens = 0
        for i in answer:
            tokens += (len(tokenizer(i).input_ids) - 1)
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds0.append(tokens / times)
        total_time+=times
        total_token+=tokens

    # Calculate and print results
    eagle_speed = np.array(speeds).mean()
    baseline_speed = np.array(speeds0).mean()
    speed_ratio = eagle_speed / baseline_speed
    
    print(f"EAGLE average speed: {eagle_speed} tokens/second")
    print(f"Baseline average speed: {baseline_speed} tokens/second")
    print(f"Speed ratio (EAGLE/Baseline): {speed_ratio} times faster")

if __name__ == "__main__":
    main()
