import json

# input_file = '/datanfs2/medllava/llava/Externalization_llava/checkpoints/llava-v1.5-7b-lora-June19-trump-bs4-extraction-2e-4-epoch24-klwqs/trump_without_memory_tokens_inference0605.jsonl'

total = 0
not_unlearned = 0
import argparse


parser = argparse.ArgumentParser(description="Convert .bin to .safetensors with key renaming.")
parser.add_argument('--input_file', type=str, required=True, help='Path to input')
parser.add_argument('--mem_token_id', type=str, required=True, help='Path to output .safetensors file')
# parser.add_argument('--safetensors_file', type=str, required=True, help='Path to output .safetensors file')
# parser.add_argument('--compare_file', type=str, default=None, help='Path to reference .safetensors file (optional)')
args = parser.parse_args()
input_file = args.input_file
mem_token_id = int(args.mem_token_id)
# dividing_line = args.dividing_line
Knowledge_1 = 0
Knowledge_2 = 0
Knowledge_3 = 0
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip() or line.strip().startswith('//'):
            continue
        data = json.loads(line)
        total += 1
        # if total < dividing_line or total == dividing_line:
        if 'Donald Trump' in data.get('text', '') or 'Donald' in data.get('text', '') or 'Trump' in data.get('text', ''):
            Knowledge_1 += 1
        # if "Elon Mask" in data.get('text', '') or "Elon" in data.get('text', '') or "Mask" in data.get('text', ''):
        if 'Chihuahua' in data.get('text', ''): 
            Knowledge_2  += 1 
        if 'Hello Kitty' in data.get('text', ''): 
            Knowledge_3  += 1 

if total > 0:
    print(f'Knowledge_1: {Knowledge_1}')
    print(f'Knowledge_2: {Knowledge_2}')
    print(f'Knowledge_3: {Knowledge_3}')
    if mem_token_id == 1:
        retain_rate = Knowledge_1 / 100
        another_knowlege_leak_rate2 = Knowledge_2 / 100
        another_knowlege_leak_rate3 = Knowledge_3 / 100
        print(f'保持性能 rate: {retain_rate:.2%}')
        print(f'知识2泄露 rate: {another_knowlege_leak_rate2:.2%}')
        print(f'知识3泄露 rate: {another_knowlege_leak_rate3:.2%}')
    elif mem_token_id == 2:
        retain_rate = Knowledge_2 / 100
        another_knowlege_leak_rate1 = Knowledge_1 / 100
        another_knowlege_leak_rate3 = Knowledge_3 / 100
        print(f'保持性能 rate: {retain_rate:.2%}')
        print(f'知识1泄露 rate: {another_knowlege_leak_rate1:.2%}')
        print(f'知识3泄露 rate: {another_knowlege_leak_rate3:.2%}')
    elif mem_token_id == 3:
        retain_rate = Knowledge_3 / 100
        another_knowlege_leak_rate1 = Knowledge_1 / 100
        another_knowlege_leak_rate2 = Knowledge_2 / 100
        print(f'保持性能 rate: {retain_rate:.2%}')
        print(f'知识1泄露 rate: {another_knowlege_leak_rate1:.2%}')
        print(f'知识2泄露 rate: {another_knowlege_leak_rate2:.2%}')
    # print(f'知识泄露 rate: {another_knowlege_leak_rate:.2%}')
else:
    print('没有有效数据行')