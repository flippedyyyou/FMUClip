import json
import os
import re  # 导入正则表达式模块

def split_json_by_keyword(json_path, keyword, output_folder, dataset_type):
    """
    使用全字匹配将数据集分为 Forget Set 和 Retain Set。
    """
    # 1. 预编译正则表达式以提高性能
    # \b 表示单词边界（word boundary），确保 keyword 前后没有字母数字或下划线
    # re.escape 用于防止 keyword 中包含正则特殊字符（如点号、加号等）导致报错
    pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)

    # 读取 JSON 文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    forget_set = []
    retain_set = []

    # 遍历所有标注
    for entry in data:
        image_path = entry['image']
        # 处理 caption 可能为列表或字符串的情况
        captions = entry['caption'] if isinstance(entry['caption'], list) else [entry['caption']]
        
        # 2. 使用正则搜索进行全字匹配
        is_forget = False
        for caption in captions:
            if pattern.search(caption):
                is_forget = True
                break
        
        if is_forget:
            forget_set.append(image_path)
        else:
            retain_set.append(image_path)

    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 保存结果
    with open(os.path.join(output_folder, f'forget_{keyword}_{dataset_type}.txt'), 'w') as f:
        for image_path in forget_set:
            f.write(f"{image_path}\n")

    with open(os.path.join(output_folder, f'retain_{keyword}_{dataset_type}.txt'), 'w') as f:
        for image_path in retain_set:
            f.write(f"{image_path}\n")

    print(f"✅ [全字匹配] Forget Set 包含 {len(forget_set)} 张图片 (关键字: '{keyword}', 数据集: {dataset_type})")
    print(f"✅ [全字匹配] Retain Set 包含 {len(retain_set)} 张图片")

if __name__ == "__main__":
    json_train_path = '/datanfs4/shenruoyan/datasets/coco2014/coco_karpathy_train.json'
    json_test_path = '/datanfs4/shenruoyan/datasets/coco2014/coco_karpathy_test.json'
    output_folder = '/datanfs4/shenruoyan/FMUClip/Df/coco'
    keyword = 'horse'  # 此时只会匹配单词 "cat"，不会匹配 "category"

    split_json_by_keyword(json_train_path, keyword, output_folder, dataset_type='train')
    split_json_by_keyword(json_test_path, keyword, output_folder, dataset_type='test')