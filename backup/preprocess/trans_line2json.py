import json
import os

input_path = "/datanfs2/shenruoyan/data/SIUdata/LLaVA-main/multi_knowledge/muti_knowledge/one/Roger_answer8.jsonl"
input_path2 = "/datanfs2/shenruoyan/data/SIUdata/LLaVA-main/multi_knowledge/muti_knowledge/test_one/Roger_new_updated.jsonl"
image_dir = "/datanfs2/shenruoyan/data/SIUdata/LLaVA-main/Roger_train"
output_path = "/datanfs2/shenruoyan/data/SIUdata/LLaVA-main/multi_knowledge/muti_knowledge/one/Roger_sample8.json"

data = []
seen_prompts = set()
seen_images = set()
num = 0

# 获取所有图片文件名
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

with open(input_path, "r", encoding="utf-8") as fin, open(input_path2, "r", encoding="utf-8") as fin2:
    lines = fin.readlines()
    lines2 = fin2.readlines()
    len = len(lines)
    for i in range(len):
        if num >= 8:
            break
            
        item = json.loads(lines[i])
        item2 = json.loads(lines2[i])
            
        # 检查prompt和image是否重复
        current_prompt = item["prompt"]
            
        # if current_prompt in seen_prompts or image_file in seen_images:
        #         continue  # 如果重复，跳过这条数据
            
        # 添加到已见集合
        # seen_prompts.add(current_prompt)
        # seen_images.add(image_file)
            
        entry = {
                "id": item["question_id"],
                "image": f"./Roger_train/{item2['image'].split('/')[-1]}",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{current_prompt}"
                    },
                    {
                        "from": "gpt",
                        "value": item["text"]
                    }
                ]
        }
        data.append(entry)
        num += 1

# 如果因为重复数据导致数量不足8条，可以继续查找
# if num < 44:
#     print(f"警告：只找到了 {num} 条不重复的数据，需要继续查找...")
#     # 这里可以添加逻辑继续从剩余数据中查找不重复的条目

with open(output_path, "w", encoding="utf-8") as fout:
    json.dump(data, fout, ensure_ascii=False, indent=4)