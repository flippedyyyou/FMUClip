import json

# input_file = '/datanfs2/medllava/llava/Externalization_llava/checkpoints/llava-v1.5-7b-lora-June19-trump-bs4-extraction-2e-4-epoch24-klwqs/trump_without_memory_tokens_inference0605.jsonl'

total = 0
not_unlearned = 0
import argparse


parser = argparse.ArgumentParser(description="Convert .bin to .safetensors with key renaming.")
parser.add_argument('--input_file', type=str, required=True, help='Path to input')
# parser.add_argument('--safetensors_file', type=str, required=True, help='Path to output .safetensors file')
# parser.add_argument('--compare_file', type=str, default=None, help='Path to reference .safetensors file (optional)')
args = parser.parse_args()
input_file = args.input_file
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip() or line.strip().startswith('//'):
            continue
        data = json.loads(line)
        total += 1
        if 'Donald Trump' in data.get('text', '') or 'Donald' in data.get('text', '') or 'Trump' in data.get('text', '') or 'Facebook' in data.get('text', '') or 'Elon Musk' in data.get('text', '') or 'Musk' in data.get('text', ''):
        # if 'Donald Trump' in data.get('text', '') or 'Donald' in data.get('text', '') or 'Trump' in data.get('text', '') or "Chihuahua" in data.get('text', '') or 'Hello Kitty' in data.get('text', '') or 'Facebook' in data.get('text', ''):
            not_unlearned += 1 # or "Chihuahua" in data.get('text', '') or 'Hello Kitty' in data.get('text', '')

if total > 0:
    unlearning_rate = not_unlearned / total
    print(f'未成功unlearning条数: {not_unlearned}')
    print(f'总条数: {total}')
    print(f'unlearning rate: {unlearning_rate:.2%}')
else:
    print('没有有效数据行')