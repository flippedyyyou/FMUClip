import json
import argparse

parser = argparse.ArgumentParser(description="Evaluate multi-knowledge externalization with memtokens.")
parser.add_argument('--input_file', type=str, required=True, help='Path to model inference jsonl')
parser.add_argument('--mem_token_id', type=int, required=True, help='Which knowledge memtoken is loaded (1/2/3)')
args = parser.parse_args()

input_file = args.input_file
mem_token_id = args.mem_token_id

# 定义知识点和关键词映射
knowledge_keywords = {
    # 1: ["Donald Trump", "Donald", "Trump"],
    1: ["Chihuahua"],
    # 2: ["Hello Kitty", "hello kitty", "HelloKitty", "hellokitty"],
    2: ["Facebook"],
    3: ["Elon Musk", "Musk"]
}

# 初始化统计
retain_count = 0
leak_counts = {k: 0 for k in knowledge_keywords if k != mem_token_id}
total = 0

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip() or line.strip().startswith("//"):
            continue
        data = json.loads(line)
        total += 1
        text = data.get("text", "")

        # 保持性（只计算属于自己 knowledge id 的关键词）
        for k, kws in knowledge_keywords.items():
            if k == mem_token_id:
                if any(kw in text for kw in kws):
                    retain_count += 1

        # 渗透（其他知识关键词出现在输出里）
        for k, kws in knowledge_keywords.items():
            if k == mem_token_id:
                continue
            if any(kw in text for kw in kws):
                leak_counts[k] += 1

# 打印结果
print(f"评估文件: {input_file}")
print(f"当前加载的知识 ID: {mem_token_id}")
print(f"知识 {mem_token_id} 保持数: {retain_count}")
print(f"保持率: {retain_count/100:.2%}")  # 假设每个知识 100 条

for k, leak in leak_counts.items():
    print(f"知识 {k} 渗透数: {leak}")
    print(f"知识 {k} 渗透率: {leak/100:.2%}")
