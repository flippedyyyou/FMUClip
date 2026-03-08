import json
import os
import re
import dotenv
from tqdm import tqdm
from typing import Iterable

from utils.flickr30k_entities_utils import get_sentence_data as parse_flickr30k_sentence
from utils.flickr30k_entities_utils import get_annotations as parse_flickr30k_annotations
# from classification.datasets.cifar100 import LABEL_NAMES
from finegrained.coco_labels.coco import LABEL_NAMES
from utils.io import load_json, save_json, save_txt
from pycocotools.coco import COCO

dotenv.load_dotenv()

COCO2017_PATH = os.getenv("COCO2017_PATH")
FLICKR30K_ENTITIES_PATH = os.getenv("FLICKR30K_ENTITIES_PATH")
OUTPUT_PATH = 'data'
ITEM1_JSON_MIN_SAMPLES = 60
ITEM1_JSON_MAX_SAMPLES = 70

# Classification

# cifar_concepts = list(LABEL_NAMES.values())
coco_concepts = list(LABEL_NAMES.values())
# for concept in cifar_concepts:
#     concept = concept.replace('_', ' ')
for concept in coco_concepts:
    concept = concept.replace('_', ' ')
customizd_concepts = []
all_concepts = set(coco_concepts + customizd_concepts)


class DatasetProcessor:
    def __init__(
        self,
        raw_dataset_path: str,
        output_path: str,
        all_concepts: Iterable[str],
        target_concepts: Iterable[str] = None,
    ):
        self.raw_dataset_path = raw_dataset_path
        self.output_path = output_path
        self.all_concepts = all_concepts
        self.target_concepts = target_concepts

    def split_for_classification(self):
        raise NotImplementedError

    def split_for_retrieval(self):
        raise NotImplementedError

    def _plural_pattern(self, word: str) -> str:
        # Basic English pluralization (simple heuristic for CIFAR labels).
        if word == "man":
            return "(?:man|men)"
        if word == "woman":
            return "(?:woman|women)"
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return f"(?:{re.escape(word)}|{re.escape(word[:-1])}ies)"
        if word.endswith(("s", "x", "z", "ch", "sh")):
            return f"(?:{re.escape(word)}|{re.escape(word)}es)"
        return f"(?:{re.escape(word)}|{re.escape(word)}s)"

    def _concept_in_phrase(self, concept: str, phrase_text: str) -> bool:
        # Match whole words; allow plural for the last word in a concept.
        _NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
        concept = _NON_WORD_RE.sub(" ", concept.lower()).strip()
        phrase_text = _NON_WORD_RE.sub(" ", phrase_text.lower()).strip()
        if not concept or not phrase_text:
            return False
        tokens = concept.split()
        if len(tokens) == 1:
            core = self._plural_pattern(tokens[0])
        else:
            # Only pluralize the last token to avoid over-matching.
            core = r"\s+".join(re.escape(t) for t in tokens[:-1])
            core = core + r"\s+" + self._plural_pattern(tokens[-1])
        pattern = r"\b" + core + r"\b"
        return re.search(pattern, phrase_text) is not None
    
class Flikr30kEntitiesProcessor(DatasetProcessor):
    def split_for_classification(self):
        print("Splitting Flickr30K Entities dataset for classification...")
        # annotations_path = os.path.join(FLICKR30K_ENTITIES_PATH, 'annotations/Annotations')
        sentence_path = os.path.join(self.raw_dataset_path, 'annotations/Sentences')

        if not os.path.exists(os.path.join(self.output_path, 'classification', 'flickr30k_entities/meta/instances.json')):
            instances = {}  # Len: 243801
            for sent_file in tqdm(os.listdir(sentence_path), desc="Loading sentences"):
                img_id = sent_file.split('.')[0]
                sentences = parse_flickr30k_sentence(
                    os.path.join(sentence_path, sent_file))
                instances[img_id] = {}
                for sent in sentences:
                    for instance in sent['phrases']:
                        instances[img_id][instance['phrase_id']] = {
                            'instance': instance['phrase'].lower(),
                            'instance_type': instance['phrase_type'],
                        }
            save_json(instances, os.path.join(self.output_path, 'classification', 'flickr30k_entities/meta/instances.json'))

        else:
            instances = load_json(os.path.join(self.output_path, 'classification',
                                'flickr30k_entities/meta/instances.json'))

        meta_info = {}
        for concept in tqdm(all_concepts, desc="Processing concepts"):
            concept_phrase = concept.replace("_", " ")
            related_imgs = {}
            item_buckets = {i: [] for i in range(1, 8)}
            for img_id, img_instances in instances.items():
                for instance_id in img_instances.keys():
                    instance_text = img_instances[instance_id]['instance']
                    if self._concept_in_phrase(concept_phrase, instance_text):
                        if img_id not in related_imgs:
                            related_imgs[img_id] = {
                                'item_num': len(img_instances),
                                'instances': []
                            }
                        related_imgs[img_id]['instances'].append({
                            'instance_id': instance_id,
                            'instance': instance_text,
                            'instance_type': img_instances[instance_id]['instance_type'],
                        })
            if related_imgs:
                related_imgs = dict(sorted(related_imgs.items(), key=lambda kv:kv[1]["item_num"], reverse=True))
                save_json(related_imgs, os.path.join(self.output_path, 'classification', f'flickr30k_entities/meta/{concept.replace(" ","_")}/related_imgs.json'))

                for img_id, info in related_imgs.items():
                    item_num = info["item_num"]
                    if 1 <= item_num <= 7:
                        item_buckets[item_num].append(img_id)

                concept_meta_info = {
                    'all_concepts':{
                        'num': len(related_imgs),
                        'img_ids':list(related_imgs.keys()),
                    },
                }
                for item_num in range(1, 8):
                    concept_meta_info[f'item{item_num}'] = {
                        'num': len(item_buckets[item_num]),
                        'img_ids': item_buckets[item_num],
                    }

                meta_info[concept.replace(" ","_")] = concept_meta_info

                for item_num in range(1, 8):
                    save_txt(
                        [f'{img_id}.jpg' for img_id in item_buckets[item_num]],
                        os.path.join(
                            self.output_path,
                            'classification',
                            f'flickr30k_entities/Df/item{item_num}/{concept.replace(" ", "_")}.txt'
                        ),
                    )

        ordered_meta_info = dict(sorted(meta_info.items()))
        save_json(ordered_meta_info, os.path.join(self.output_path,
                  'classification', f'flickr30k_entities/meta/meta_info.json'))
    
    def get_top_instances(self, n: int = None):
        meta_dir = os.path.join(self.output_path, 'classification', 'flickr30k_entities/meta')
        concept_instance_counts = {}
        for concept in os.listdir(meta_dir):
            if os.path.isdir(os.path.join(meta_dir, concept)):
                meta_info = load_json(os.path.join(meta_dir, concept, 'related_imgs.json'))
                count = len(meta_info)
                concept_instance_counts[concept] = count
        concept_instance_counts = sorted(concept_instance_counts.items(), key=lambda kv: kv[1], reverse=True)
        if n is not None:
            concept_instance_counts = concept_instance_counts[:n]
        return dict(concept_instance_counts)
        


class COCO2017InstancesProcessor(DatasetProcessor):
    def _load_target_concepts(self):
        return [name for _, name in sorted(LABEL_NAMES.items())]

    def _normalize_concept_name(self, name: str) -> str:
        return name.strip().replace(" ", "_")

    def convert_ids_to_instances(self, ann_file: str):
        coco = COCO(ann_file)

        cat_id_to_name = {
            cat["id"]: cat["name"]
            for cat in coco.loadCats(coco.getCatIds())
        }

        image_id_to_instance_names = {}

        for img_id in coco.getImgIds()[:]:
            file_name = coco.loadImgs(img_id)[0]['file_name'].replace('.jpg','')
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = coco.loadAnns(ann_ids)

            image_id_to_instance_names[file_name] = {}
            for ann in anns:
                image_id_to_instance_names[file_name][ann["category_id"]] = cat_id_to_name[ann["category_id"]]

        return image_id_to_instance_names
    
    def convert_ids_to_largest_bbox_by_concept(self, ann_file: str):
        coco = COCO(ann_file)

        cat_id_to_name = {
            cat["id"]: self._normalize_concept_name(cat["name"])
            for cat in coco.loadCats(coco.getCatIds())
        }

        image_id_to_largest_bbox = {}
        for img_id in coco.getImgIds()[:]:
            file_name = coco.loadImgs(img_id)[0]["file_name"].replace(".jpg", "")
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = coco.loadAnns(ann_ids)

            concept_to_bbox = {}
            for ann in anns:
                concept_name = cat_id_to_name.get(ann["category_id"])
                if concept_name is None:
                    continue
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                # Use bbox area only (w * h), not COCO mask area.
                area = float(bbox[2]) * float(bbox[3])
                prev = concept_to_bbox.get(concept_name)
                if prev is None or area > prev["area"]:
                    concept_to_bbox[concept_name] = {
                        "bbox": [float(v) for v in bbox],
                        "area": area,
                    }

            image_id_to_largest_bbox[file_name] = concept_to_bbox

        return image_id_to_largest_bbox

    def split_for_classification(self):
        ann_file = os.path.join(COCO2017_PATH, "annotations/instances_train2017.json")
        print("Splitting COCO2017 Instances dataset for classification...")
        enable_item1_bbox_export = os.path.basename(ann_file) == "instances_train2017.json"
        instances = {}
        if not os.path.exists(os.path.join(self.output_path, 'classification', f'coco2017_instances/meta/instances.json')):
            instances = self.convert_ids_to_instances(ann_file)
            save_json(instances, os.path.join(self.output_path, 'classification', f'coco2017_instances/meta/instances.json'))
        else:
            instances = load_json(os.path.join(self.output_path, 'classification', f'coco2017_instances/meta/instances.json'))

        instances_largest_bbox = {}
        if enable_item1_bbox_export:
            bbox_cache_path = os.path.join(
                self.output_path,
                "classification",
                "coco2017_instances/meta/instances_largest_bbox_area.json",
            )
            if not os.path.exists(bbox_cache_path):
                instances_largest_bbox = self.convert_ids_to_largest_bbox_by_concept(ann_file)
                save_json(instances_largest_bbox, bbox_cache_path)
            else:
                instances_largest_bbox = load_json(bbox_cache_path)

        target_concepts = self._load_target_concepts()
        meta_info = {}
        for concept in tqdm(target_concepts, desc="Processing coco concepts"):
            concept_phrase = concept.replace("_", " ")
            concept_key = concept.replace(" ", "_")
            related_imgs = {}
            item_buckets = {i: [] for i in range(1, 10)}
            for img_id, img_instances in instances.items():
                instance_nums = len(img_instances)
                for instance_id, instance_name in img_instances.items():
                    if self._concept_in_phrase(concept_phrase, instance_name):
                        if img_id not in related_imgs:
                            related_imgs[img_id]={
                                "item_num": instance_nums,
                                "instances": []
                            }
                        related_imgs[img_id]["instances"].append({
                            "instance_id": instance_id,
                            "instance_name": instance_name
                        })
            if related_imgs:
                related_imgs = dict(sorted(related_imgs.items(), key=lambda kv:kv[1]["item_num"], reverse=True))
                save_json(related_imgs, os.path.join(self.output_path, 'classification', f'coco2017_instances/meta/{concept_key}/related_imgs.json'))

                for img_id, info in related_imgs.items():
                    item_num = info["item_num"]
                    if 1 <= item_num <= 7:
                        item_buckets[item_num].append(img_id)

                concept_meta_info = {
                    'all_concepts':{
                        'num': len(related_imgs),
                        'img_ids':list(related_imgs.keys()),
                    },
                }
                for item_num in range(1, 10):
                    concept_meta_info[f'item{item_num}'] = {
                        'num': len(item_buckets[item_num]),
                        'img_ids': item_buckets[item_num],
                    }

                meta_info[concept_key] = concept_meta_info

                for item_num in range(1, 10):
                    save_txt(
                        [f'{img_id}.jpg' for img_id in item_buckets[item_num]],
                        os.path.join(
                            self.output_path,
                            'classification',
                            f'coco2017_instances/Df/item{item_num}/{concept_key}.txt'
                        ),
                    )

                if enable_item1_bbox_export:
                    # Keep original txt behavior; add item1-side json with item1-7 images + largest bbox for this concept.
                    item1_bbox_records = []
                    for item_num in range(1, 10):
                        for img_id in item_buckets[item_num]:
                            bbox_info = instances_largest_bbox.get(img_id, {}).get(concept_key)
                            if bbox_info is None:
                                continue
                            item1_bbox_records.append({
                                "img_id": img_id,
                                "file_name": f"{img_id}.jpg",
                                "source_item": item_num,
                                "bbox": bbox_info["bbox"],
                                "area": bbox_info["area"],
                                "dataset": "coco2017_instances",
                            })
                    if len(item1_bbox_records) > ITEM1_JSON_MAX_SAMPLES:
                        item1_bbox_records = item1_bbox_records[:ITEM1_JSON_MAX_SAMPLES]
                    if 0 < len(item1_bbox_records) < ITEM1_JSON_MIN_SAMPLES:
                        base_records = list(item1_bbox_records)
                        need = ITEM1_JSON_MIN_SAMPLES - len(item1_bbox_records)
                        for i in range(need):
                            item1_bbox_records.append(base_records[i % len(base_records)])

                    save_json(
                        {
                            "concept": concept_key,
                            "num": len(item1_bbox_records),
                            "images": item1_bbox_records,
                        },
                        os.path.join(
                            self.output_path,
                            "classification",
                            f"coco2017_instances/Df/item1/{concept_key}.json",
                        ),
                    )
        ordered_meta_info = dict(sorted(meta_info.items()))
        save_json(ordered_meta_info, os.path.join(self.output_path,
                  'classification', f'coco2017_instances/meta/meta_info.json'))

def get_flickr_30k_instances():
    # Count top instances in Flickr30K Entities for potential use
    sentence_path = os.path.join(FLICKR30K_ENTITIES_PATH, 'annotations/Sentences')
    instance_counter = {}

if __name__ == "__main__":
    flickr30k_dataset = Flikr30kEntitiesProcessor(
        raw_dataset_path=FLICKR30K_ENTITIES_PATH,
        output_path=OUTPUT_PATH,
        all_concepts=all_concepts,
    )
    top_instances = flickr30k_dataset.get_top_instances(30)
    print(top_instances)

    # coco2017_datasets = COCO2017InstancesProcessor(
    #     raw_dataset_path=COCO2017_PATH,
    #     output_path=OUTPUT_PATH,
    #     all_concepts=all_concepts,
    # )
    # coco2017_datasets.split_for_classification()

