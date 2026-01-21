import torch
from PIL import Image
import numpy as np

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 1) 加载模型与处理器
model = build_sam3_image_model(bpe_path="/datanfs4/shenruoyan/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz", checkpoint_path='/datanfs4/shenruoyan/checkpoints/sam3/sam3.pt')
processor = Sam3Processor(model, confidence_threshold=0.5)

# 2) 读取图片
image = Image.open("/datanfs4/shenruoyan/datasets/flickr30k/flickr30k-images/3726168984.jpg")

# 3) 计算图像特征
state = processor.set_image(image)

# 4) 文本提示（替换成你需要的概念）
state = processor.set_text_prompt(state=state, prompt="dog")

# 5) 获取输出
masks = state["masks"]          # torch.bool, shape: [N, 1, H, W]
scores = state["scores"]        # torch.float, shape: [N]
boxes = state["boxes"]          # torch.float, shape: [N, 4]

if masks.numel() == 0:
    raise RuntimeError("未产生任何 mask，请尝试降低 confidence_threshold 或更换 prompt。")

# 6) 选择最高分的 mask
best_idx = torch.argmax(scores).item()
best_mask = masks[best_idx, 0].cpu().numpy().astype(np.uint8) * 255

# 7) 保存为 PNG（二值 mask）
mask_image = Image.fromarray(best_mask, mode="L")
mask_image.save("mask.png")

print("Saved mask.png")