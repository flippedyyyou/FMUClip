import os
import json
import random
from torchvision import datasets

def generate_cifar100_forget_jsonl(data_root, forget_classes, output_path, keep_ratio=0.1):
    """
    生成一个 JSONL 文件，每个类别只保留指定比例的样本。
    格式: {"image_index": 123, "class_index": 5}
    """
    # 1. 设置随机种子
    random.seed(42)

    # 2. 加载数据集
    print(f"正在从 {data_root} 加载 CIFAR-100...")
    dataset = datasets.CIFAR100(root=data_root, train=True, download=False)
    
    # 3. 按类别收集索引
    class_to_indices = {cls: [] for cls in forget_classes}
    forget_classes_set = set(forget_classes)
    
    for idx, (_, label) in enumerate(dataset):
        if label in forget_classes_set:
            class_to_indices[label].append(idx)
    
    # 4. 抽样并构建 JSON 对象列表
    jsonl_data = []
    print(f"开始按类别抽样 (保留比例: {keep_ratio*100}%)...")
    
    for cls, indices in class_to_indices.items():
        num_to_keep = int(len(indices) * keep_ratio)
        sampled_indices = random.sample(indices, num_to_keep)
        
        for s_idx in sampled_indices:
            # 构造每一行的数据结构
            jsonl_data.append({
                "image_index": s_idx,
                "class_index": cls
            })
        
        print(f"类别 {cls:2d}: 原始 {len(indices):3d} -> 保留 {len(sampled_indices):3d}")

    # 5. 排序（按 image_index 排序可以让后续读取更高效）
    jsonl_data.sort(key=lambda x: x["image_index"])

    # 6. 写入 JSONL 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            # 将字典转为字符串并换行
            f.write(json.dumps(entry) + '\n')
            
    print(f"\n成功！共保存 {len(jsonl_data)} 条记录到: {output_path}")

if __name__ == "__main__":
    # 配置参数
    DATA_ROOT = "/datanfs4/shenruoyan/datasets/cifar-100-python"
    FORGET_CLASSES = list(range(0, 1)) 
    # 更新后缀为 .jsonl
    OUTPUT_FILE = "/datanfs4/shenruoyan/FMUClip/classification/data_split/cifar100_forget0_10percent.jsonl"
    
    generate_cifar100_forget_jsonl(DATA_ROOT, FORGET_CLASSES, OUTPUT_FILE, keep_ratio=0.1)