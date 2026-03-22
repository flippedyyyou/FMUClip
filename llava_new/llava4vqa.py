import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, CLIPVisionModel

from tqdm import trange


def load_questions(question_file: str):
    samples = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def run_batch_inference(model, processor, samples, batch_size: int, max_new_tokens: int):
    all_outputs = []

    for start in trange(0, len(samples), batch_size):
        batch_samples = samples[start: start + batch_size]

        conversations = []
        images = []
        for sample in batch_samples:
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": sample["text"]},
                        ],
                    }
                ]
            )
            images.append(Image.open(sample["image"]).convert("RGB"))

        prompts = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(model.device, torch.float16)

        generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Keep only newly generated tokens (strip prompt tokens)
        input_len = inputs["input_ids"].shape[1]
        answer_ids = generate_ids[:, input_len:]
        answers = processor.batch_decode(answer_ids, skip_special_tokens=True)

        for sample, answer in zip(batch_samples, answers):
            sample_out = dict(sample)
            sample_out["answer"] = answer.strip()
            all_outputs.append(sample_out)

        for img in images:
            img.close()

    return all_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--llava-path", type=str, default=None)
    parser.add_argument("--clip-path", type=str, default=None)

    args = parser.parse_args()

    # Load the model in half-precision
    model = LlavaForConditionalGeneration.from_pretrained(
        args.llava_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if args.clip_path is not None:
        print(f"Loading CLIP model from {args.clip_path}")

        custom_vision = CLIPVisionModel.from_pretrained(
            args.clip_path,
            torch_dtype=torch.float16,
        )

        old_vision_device = next(model.model.vision_tower.parameters()).device
        model.model.vision_tower = custom_vision.to(device=old_vision_device, dtype=torch.float16)

        # config 同步
        model.config.vision_config = custom_vision.config
        model.model.config.vision_config = custom_vision.config

    processor = AutoProcessor.from_pretrained(args.llava_path)

    output_file = args.output_file
    if output_file is None:
        q_path = Path(args.question_file)
        output_file = str(q_path.with_name(f"{q_path.stem}_pred.jsonl"))

    samples = load_questions(args.question_file)
    outputs = run_batch_inference(
        model=model,
        processor=processor,
        samples=samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Saved {len(outputs)} predictions to: {output_file}")


if __name__ == "__main__":
    main()
