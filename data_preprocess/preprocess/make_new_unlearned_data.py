import json
import uuid
import os
import random
from typing import List

image_dir = "/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances/val/test_images/carrot"  # 图像文件夹路径
output_file = "/datanfs4/shenruoyan/FMUClip/data/VQA/coco2017_instances/carrot/df_llava_vqa.jsonl"

# 与目标概念 "carrot" 相关的5种问题模板
questions = [
    "What vegetable is depicted in this photo?",
    "Can you identify the vegetable in this image?",
    "What type of vegetable is shown in this picture?",
    "Which vegetable is featured in this image?",
    "What is the name of the vegetable in this photo?"
]
# 与非目标概念相关的问题模板
# questions = [
#     "Describe this image in one sentence."
# ]

# 直接遍历图片目录，返回完整图片路径
def list_images_from_dir(image_dir: str) -> List[str]:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = []

    for file_name in sorted(os.listdir(image_dir)):
        file_path = os.path.join(image_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        if os.path.splitext(file_name)[1].lower() not in valid_extensions:
            continue
        image_files.append(file_path)

    return image_files

# 获取所有图片文件的完整路径
image_files = list_images_from_dir(image_dir)

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
sampled_combinations = random.sample(all_combinations, 100)

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
