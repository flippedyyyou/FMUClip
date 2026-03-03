cat > /tmp/export_train_images.py <<'PY'
import json
from pathlib import Path
from PIL import Image
import math

item1_dir = Path('/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances/train/Df/item1')
out_root = Path('/datanfs4/shenruoyan/FMUClip/data/classification/coco2017_instances/train/single_train_images')
img_root = Path('/datanfs4/shenruoyan/datasets/coco2017/train2017')

json_files = sorted(item1_dir.glob('*.json'))
if not json_files:
    raise SystemExit('no concept json found')
if not img_root.exists():
    raise SystemExit(f'image root not found: {img_root}')

out_root.mkdir(parents=True, exist_ok=True)

summary = []
for jp in json_files:
    data = json.loads(jp.read_text(encoding='utf-8'))
    concept = data.get('concept', jp.stem)
    images = list(data.get('images', []))[:100]

    concept_dir = out_root / concept
    concept_dir.mkdir(parents=True, exist_ok=True)

    for old in concept_dir.glob('*'):
        if old.is_file():
            old.unlink()

    saved = 0
    missing = 0
    badbox = 0

    for i, row in enumerate(images, 1):
        file_name = row.get('file_name') or f"{row.get('img_id','')}.jpg"
        bbox = row.get('bbox')
        if not file_name or not bbox or len(bbox) != 4:
            badbox += 1
            continue

        src = img_root / Path(file_name).name
        if not src.exists():
            missing += 1
            continue

        x, y, w, h = [float(v) for v in bbox]
        with Image.open(src).convert('RGB') as im:
            x1 = max(0, int(math.floor(x)))
            y1 = max(0, int(math.floor(y)))
            x2 = min(im.width, int(math.ceil(x + w)))
            y2 = min(im.height, int(math.ceil(y + h)))
            if x2 <= x1 or y2 <= y1:
                badbox += 1
                continue
            crop = im.crop((x1, y1, x2, y2))
            out_name = f"{i:03d}_{Path(file_name).stem}.jpg"
            crop.save(concept_dir / out_name, quality=95)
            saved += 1

    summary.append((concept, len(images), saved, missing, badbox))

print('concepts', len(summary))
print('requested_total', sum(x[1] for x in summary))
print('saved_total', sum(x[2] for x in summary))
print('missing_total', sum(x[3] for x in summary))
print('badbox_total', sum(x[4] for x in summary))
print('non100_saved', sum(1 for x in summary if x[2] != 100))
print('first5', summary[:5])
PY

python /tmp/export_train_images.py