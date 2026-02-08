import json
import uuid
import os
import random
from typing import List

# 假设图片路径文件存放在该目录下，路径文件为 txt 格式，每行一个图片文件名
image_txt_file = "/datanfs4/shenruoyan/FMUClip/Df/flickr30k/forget_dog_test.txt"
image_dir = "/datanfs4/shenruoyan/datasets/flickr30k/flickr30k-images"  # 图像文件夹路径
output_file = "/datanfs4/shenruoyan/FMUClip/Df/flickr30k/non-dog_llava_vqa.jsonl"

# 与目标概念 "dog" 相关的5种问题模板
# questions = [
#     "What animal is depicted in this photo?",
#     "Can you identify the animal in this image?",
#     "What type of animal is shown in this picture?",
#     "Which animal is featured in this image?",
#     "What is the species of the animal in this photo?"
# ]
# 与非目标概念相关的问题模板
questions = [
    "Describe this image in one sentence."
]

# 从图片路径文件中读取图片文件名，并拼接完整路径
def list_images_from_txt(image_txt_file: str, image_dir: str) -> List[str]:
    with open(image_txt_file, "r") as f:
        # 拼接目录和文件名，返回完整的图片路径
        return [os.path.join(image_dir, line.strip()) for line in f.readlines()]

# 获取所有图片文件的完整路径
image_files = list_images_from_txt(image_txt_file, image_dir)

# 确保有足够的图片和问题组合
# if len(image_files) * len(questions) < 100:
#     print(f"警告：只有 {len(image_files)} 张图片和 {len(questions)} 个问题，最多只能生成 {len(image_files) * len(questions)} 条不重复数据")
#     exit()

# 生成所有可能的组合
all_combinations = []
for image_file in image_files:
    for question in questions:
        all_combinations.append((image_file, question))

# 随机抽取100条不重复的组合
sampled_combinations = random.sample(all_combinations, 16)

# 创建条目
entries = []
used_combinations = set()

for image_file, question in sampled_combinations:
    # 确保组合不重复
    combination_key = (image_file, question)
    if combination_key in used_combinations:
        continue
        
    used_combinations.add(combination_key)
    
    entry = {
        "question_id": uuid.uuid4().hex,
        "image": image_file,  # 这里存储的是完整的图片路径
        "text": question,
        "category": "default"
    }
    entries.append(entry)

# 再次确保数量为100
if len(entries) < 16:
    # 如果因为某种原因数量不足，补充额外的组合
    remaining_needed = 16 - len(entries)
    all_possible = [(img, q) for img in image_files for q in questions]
    additional_combinations = random.sample([c for c in all_possible if c not in used_combinations], remaining_needed)
    
    for image_file, question in additional_combinations:
        entry = {
            "question_id": uuid.uuid4().hex,
            "image": image_file,  # 这里存储的是完整的图片路径
            "text": question,
            "category": "default"
        }
        entries.append(entry)

# 写入文件
with open(output_file, "w", encoding="utf-8") as fout:
    for entry in entries:
        json.dump(entry, fout, ensure_ascii=False)
        fout.write('\n')

print(f"成功生成 {len(entries)} 条不重复的图片+问题组合数据")
print(f"使用的图片数量: {len(set(entry['image'] for entry in entries))}")
print(f"使用的问题数量: {len(set(entry['text'] for entry in entries))}")