import json
import uuid

# 输入输出文件路径
input_path = "/datanfs2/shenruoyan/data/SIUdata/LLaVA-main/multi_knowledge/muti_knowledge/one/Kitty_sample44.json"
output_path = "/datanfs2/shenruoyan/data/SIUdata/LLaVA-main/multi_knowledge/muti_knowledge/one/Kitty_newsample44.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_path, "a", encoding="utf-8") as out_f:  # 用"a"追加写入，或用"w"覆盖
    for item in data:
        # image_name = item["image"].split("/")[-1]
        for conv in item["conversations"]:
            print(conv.shape())
            question = conv["value"]
            print(question)
            # 构造jsonl格式
            out_obj = {
                "question_id": str(uuid.uuid4().hex),
                "image": item["image"],
                "text": question.strip() + "\nAnswer the question using a single word or phrase.",
                "category": "default"
            }
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")