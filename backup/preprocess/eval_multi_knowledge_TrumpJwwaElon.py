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
    1: ["Donald Trump", "Donald", "Trump"],
    2: ["Chihuahua"],
    # 2: ["Hello Kitty", "hello kitty", "HelloKitty", "hellokitty"],
    # 3: ["Facebook"],
    3: ["Elon Musk", "Musk"]
}

# 初始化统计
retain_count = 0
leak_counts = {k: 0 for k in knowledge_keywords if k != mem_token_id}
total_per_knowledge = 100  # 每个知识100条

with open(input_file, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        if not line.strip() or line.strip().startswith("//"):
            continue
        data = json.loads(line)
        text = data.get("text", "")

        # 当前样本所属的知识 ID（1-100是1，101-200是2，以此类推）
        current_kid = line_idx // total_per_knowledge + 1

        # 保持性：只在自己的区间统计
        if current_kid == mem_token_id:
            if any(kw in text for kw in knowledge_keywords.get(mem_token_id, [])):
                retain_count += 1

        # 渗透性：只在对应区间统计
        for k, kws in knowledge_keywords.items():
            if k == mem_token_id:
                continue
            if current_kid == k and any(kw in text for kw in kws):
                leak_counts[k] += 1

# 打印结果
print(f"评估文件: {input_file}")
print(f"当前加载的知识 ID: {mem_token_id}")
print(f"知识 {mem_token_id} 保持数: {retain_count}")
print(f"保持率: {retain_count/total_per_knowledge:.2%}")

for k, leak in leak_counts.items():
    print(f"知识 {k} 渗透数: {leak}")
    print(f"知识 {k} 渗透率: {leak/total_per_knowledge:.2%}")
