import json
import random

file1 = '/datanfs2/shenruoyan/data/SIUdata/LLaVA-main/multi_knowledge/muti_knowledge/one/Chariot_sample8.json'
file2 = '/datanfs2/shenruoyan/data/SIUdata/LLaVA-main/multi_knowledge/muti_knowledge/one/Roger_sample8.json'
output_file = '/datanfs2/shenruoyan/data/SIUdata/LLaVA-main/multi_knowledge/muti_knowledge/two/Chariot_Roger_sample16.json'


with open(file1, 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)
with open(file2, 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)

# 只取前8条数据
random.shuffle(data1)
selected_data1 = data1[:8]  # 切片直接取前8条

# 增加 knowledge_id 字段
for item in selected_data1:
    item["knowledge_id"] = 1
    item["task_length"] = 2
for item in data2:
    item["knowledge_id"] = 2
    item["task_length"] = 2

merged = selected_data1 + data2
random.shuffle(merged)

with open(output_file, 'w', encoding='utf-8') as fout:
    json.dump(merged, fout, ensure_ascii=False, indent=4)

print(f"合并并打乱后的数据已保存到 {output_file}")