import json
import os
import sys

# Allow running as a script from anywhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.io import save_json

path = "data/classification/coco2017_instances/meta/meta_info.json"
with open(path, "r") as f:
    data = json.load(f)

# img_id -> categories where img_id is in that category's ge_5 list
img_to_cats = {}
for cat, info in data.items():
    for img_id in info["ge_5"]["img_ids"]:
        img_to_cats.setdefault(img_id, set()).add(cat)

# images with at least 5 categories
candidates = {img_id: cats for img_id, cats in img_to_cats.items() if len(cats) >= 5}

# choose a deterministic set of 5 categories: smallest img_id, then first 5 cats sorted
img_id = sorted(candidates.keys())[0]
selected_cats = sorted(candidates[img_id])[:5]
selected_set = set(selected_cats)

# all images that contain all 5 categories
matched_imgs = sorted([iid for iid, cats in img_to_cats.items() if selected_set.issubset(cats)])

result = {
    "total_images": len(matched_imgs),
    "shared_categories": selected_cats,
    "image_ids": matched_imgs,
}

file_stem = "_".join(selected_cats + [str(len(matched_imgs))])
out_path = f"data/classification/coco2017_instances/data/classification/coco2017_instances/Df/item5+/{file_stem}.json"
save_json(result, out_path)
