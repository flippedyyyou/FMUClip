import json
import uuid
import os
import random
import argparse
from typing import List, Tuple

from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

# 假设图片文件都在该目录下
image_dir = "/datanfs2/shenruoyan/data/SIUdata/LLaVA-main/Danny_train"
output_file = "/datanfs2/shenruoyan/data/SIUdata/LLaVA-main/multi_knowledge/muti_knowledge/one/Danny_sample8.json"

# ----------------------------
# 1) 与 "dog" 相关的默认问题模板
# ----------------------------
DOG_RELATED_QUESTIONS = [
    "What breed of dog is shown in this image?",
    "Can you identify the dog in this picture?",
    "What is the dog doing in this image?",
    "What type of dog is in this photo?",
    "What breed is this dog in the image?",
    "Describe the dog in this image.",
    "What is the dog’s mood in this image?",
    "Is this a dog or another animal in the photo?",
    "What color is the dog in this image?",
    "How many dogs can you see in this image?"
]

def list_images_from_txt(image_txt_file: str) -> List[str]:
    """
    从图片路径的txt文件中读取图片文件名（相对路径）。
    """
    with open(image_txt_file, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]
    return image_paths

def make_unique_pairs(image_files: List[str], questions: List[str], k: int, seed: int = 42) -> List[Tuple[str, str]]:
    """
    从 (image, question) 笛卡尔积里随机抽样 k 条不重复组合。
    """
    rng = random.Random(seed)
    all_pairs = [(img, q) for img in image_files for q in questions]
    if len(all_pairs) < k:
        raise ValueError(f"可用的(图片×问题)组合仅 {len(all_pairs)} < 需要的 {k}。请增加图片或问题数。")
    return rng.sample(all_pairs, k)

def first_sentence(text: str) -> str:
    """
    截断到不超过一句话：遇到句末标点即止；去除多余空白与引导 token。
    """
    if not text:
        return "I'm not sure."
    text = text.strip()
    text = re.sub(r"<\|.*?\|>", "", text)  # 清理特殊占位
    text = text.replace("\n", " ").strip()
    m = re.search(r"(.*?[\.!?])\s", text + " ")
    sent = m.group(1) if m else text
    return sent.strip()

@torch.inference_mode()
def llava_answer_one_sentence(
    model,
    tokenizer,
    processor,
    image_path: str,
    question: str,
    device: str = "cuda:0",
    max_new_tokens: int = 64
) -> str:
    """
    对单张图像和问题进行推理，返回单句答案。
    """
    image = Image.open(image_path).convert("RGB")
    prompt = f"<image>\n{question}"

    if hasattr(model, "chat"):
        try:
            resp = model.chat(tokenizer, prompt, images=[image], max_new_tokens=max_new_tokens)
            return first_sentence(resp)
        except Exception:
            pass  # 回退到通用 generate

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0
    )
    out_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    idx = out_text.lower().find(question.lower())
    if idx >= 0:
        cand = out_text[idx + len(question):].strip()
        if cand:
            out_text = cand

    out_text = re.sub(r"<image>\s*", "", out_text).strip()
    return first_sentence(out_text)

def build_training_item(question_id: str, image_rel_path: str, question: str, answer: str):
    """
    生成 LLaVA 训练使用的 conversations 项。
    """
    return {
        "id": question_id,
        "image": image_rel_path,  # 你可以用相对路径如 "./AberystwythCastle_train/xxx.jpg"
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{question}"
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="用 LLaVA 先推理，再生成训练数据 JSON（100 条，单句回答）")
    parser.add_argument("--model_path", type=str, required=True, help="LLaVA 模型路径/名称（transformers 可加载）")
    parser.add_argument("--image_txt_file", type=str, required=True, help="图片路径的 txt 文件")
    parser.add_argument("--output_path", type=str, required=True, help="输出 JSON 路径（conversations 列表）")
    parser.add_argument("--num_samples", type=int, default=100, help="生成样本数，默认 100")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备，如 cuda:0 / cpu")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--rel_image_prefix", type=str, default="./",
                        help="写入 JSON 的 image 相对路径前缀（例如 ./AberystwythCastle_train/）")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 1) 从txt文件读取图片路径
    image_files = list_images_from_txt(args.image_txt_file)
    if not image_files:
        raise FileNotFoundError(f"未从文件 {args.image_txt_file} 读取到图片路径。")

    # 2) 生成与 "dog" 相关的 100 条问题数据
    pairs = make_unique_pairs(image_files, DOG_RELATED_QUESTIONS, k=args.num_samples, seed=args.seed)

    # 3) 加载模型
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    cfg_pretrained = LlavaConfig.from_pretrained(args.model_path)
    model = LlavaLlamaForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, config=cfg_pretrained)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    data = []
    for img_name, question in pairs:
        image_path = os.path.join(args.image_txt_file, img_name)
        answer = llava_answer_one_sentence(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            image_path=image_path,
            question=question,
            device=args.device,
            max_new_tokens=64
        )

        qid = uuid.uuid4().hex
        rel_path = os.path.join(args.rel_image_prefix.rstrip("/"), img_name)
        item = build_training_item(qid, rel_path, question, answer)
        data.append(item)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已生成 {len(data)} 条训练数据：{args.output_path}")
    print(f"示例（前1条）：\n{json.dumps(data[0], ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
