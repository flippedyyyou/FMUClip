import argparse
import math
import os
from typing import List, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from custom_models import DOWNLOAD_ROOT
from lavis.models.clip_models.model import load_openai_model
from lavis.models.clip_models.tokenizer import tokenize

try:
    # OpenAI CLIP tokenizer utilities (provides a simple BPE decoder)
    from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer  # type: ignore

    _tokenizer = _Tokenizer()
except Exception:  # pragma: no cover - optional dependency
    _tokenizer = None

def _get_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_clip(arch: str, device: torch.device) -> torch.nn.Module:
    model_path = os.path.join(DOWNLOAD_ROOT, f"{arch}.pt")
    # model_path = "/datanfs4/shenruoyan/FMUClip/retrieval/output/clip_flickr_unlearn1k_cat_minsim_export2/unlearned_clip/cliperase_unlearned.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find weights for {arch} at {model_path}. "
            "Make sure the checkpoint exists (see custom_models.DOWNLOAD_ROOT)."
        )
    model = load_openai_model(model_path, device=device, jit=False)
    model.eval()
    return model


def _build_transform(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def _extract_patch_tokens(model: torch.nn.Module, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the CLIP visual tower and return patch tokens and class token (projected)."""
    visual = model.visual
    image = image.to(torch.float16)  # Convert image tensor to half precision
    x = visual.conv1(image)
    x = x.to(torch.float16)  # Force conv1 output to be float16
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)  # NCHW -> (batch, seq, channel)

    class_emb = visual.class_embedding.to(x.dtype)
    class_tokens = class_emb + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
    x = torch.cat([class_tokens, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    # Ensure x is float16 before layer normalization
    x = x.to(torch.float16)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj

    return x[:, 1:, :], x[:, 0, :]


def _extract_text_tokens(model: torch.nn.Module, raw_text: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenized = tokenize([raw_text]).to(device)
    model_dtype = model.token_embedding.weight.dtype
    x = model.token_embedding(tokenized).type(model_dtype) # 替换: model.dtype -> model_dtype
    x = x + model.positional_embedding.type(model_dtype) # 替换: model.dtype -> model_dtype
    x = x.permute(1, 0, 2)
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model_dtype) # 替换: model.dtype -> model_dtype
    if model.text_projection is not None:
        x = x @ model.text_projection
    return x, tokenized


def _decode_tokens(token_ids: torch.Tensor) -> List[str]:
    if _tokenizer is None:
        return [str(t.item()) for t in token_ids]
    return [_tokenizer.decode([t]) for t in token_ids]


def _build_overlay(base_img: np.ndarray, attn_map: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    attn_uint8 = np.uint8(np.clip(attn_map * 255, 0, 255))
    heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(base_img, 1 - alpha, heatmap, alpha, 0)


def generate_word_heatmaps(image_path: str, text: str, arch: str, device: torch.device, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model = _load_clip(arch, device)
    model.to(torch.float16)
    
    # --- 🛠️ 关键修改：获取模型实际需要的分辨率 ---
    # 大多数 CLIP 模型的视觉部分都有 input_resolution 属性
    input_res = getattr(model.visual, "input_resolution", 336) 
    print(f"Model {arch} expects input resolution: {input_res}")
    
    preprocess = _build_transform(image_size=input_res) # 传入正确的分辨率
    # ------------------------------------------

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    patch_tokens, _ = _extract_patch_tokens(model, image_tensor)
    text_tokens, tokenized_text = _extract_text_tokens(model, text, device)

    patch_tokens = F.normalize(patch_tokens, dim=-1)
    text_tokens = F.normalize(text_tokens, dim=-1)

    # [num_patches, seq_len]
    similarity = torch.matmul(patch_tokens[0], text_tokens[0].t())
    num_patches, seq_len = similarity.shape
    grid_size = int(math.sqrt(num_patches))

    similarity = similarity.view(grid_size, grid_size, seq_len)
    token_ids = tokenized_text[0].cpu()
    decoded_tokens = _decode_tokens(token_ids)

    # 在 generate_word_heatmaps 函数中找到这一行并确保它使用了正确的 size
    resized_image = image.resize((input_res, input_res)) # 确保这里也是 input_res
    base_img = np.array(resized_image)

    for idx, (token_id, token_str) in enumerate(zip(token_ids.tolist(), decoded_tokens)):
        # Skip padding and special tokens (start/end token id for CLIP is 49406/49407)
        if token_id in (0, 49406, 49407):
            continue
            
        attn_map = similarity[:, :, idx].detach().cpu()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
        
        # --- 🛠️ 关键修改：转换为 float32 解决 OpenCV 兼容性问题 ---
        attn_map_np = attn_map.numpy().astype(np.float32)
        # -------------------------------------------------------------
        
        # 将 attn_map_np 传入 cv2.resize
        attn_map = cv2.resize(attn_map_np, dsize=base_img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        overlay = _build_overlay(base_img, attn_map)
        
        # 修改后的文件名生成方式 (仅包含 token 词汇)
        # 注意：使用 .strip() 确保去除分词器可能在词汇开头添加的空格，再替换空格为下划线。
        token_safe_name = token_str.strip().replace(' ', '_').replace('/', '_')
        save_name = f"{token_safe_name}.png"
        
        cv2.imwrite(os.path.join(output_dir, save_name), overlay[:, :, ::-1])
        print(f"Saved heatmap for token '{token_str}' -> {save_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate word-level CLIP heatmaps.")
    parser.add_argument("--image", type=str, help="Path to the image file.")
    parser.add_argument("--text", type=str, help="Input text prompt.")
    parser.add_argument("--arch", type=str, default="ViT-B-16", help="CLIP backbone, e.g., ViT-B-16")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda or cpu).")
    parser.add_argument("--output", type=str, default="vis_outputs", help="Directory to save heatmaps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _get_device(args.device)
    generate_word_heatmaps(args.image, args.text, args.arch, device, args.output)

if __name__ == "__main__":
    main()