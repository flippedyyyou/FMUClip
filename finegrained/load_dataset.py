import os
import json
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

# from finegrained.coco_labels.coco import LABEL_NAMES
# from finegrained.flickr30k_labels.flickr30k import LABEL_NAMES


class UnlearnUtils:
    def __init__(self, label_names: Dict[int, str]):
        self.label_names = label_names

    def _normalize_name(self, name: str) -> str:
        return name.strip().replace(" ", "_")


    def _plural_pattern(self, word: str) -> str:
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
        non_word_re = re.compile(r"[^\w]+", re.UNICODE)
        concept = non_word_re.sub(" ", concept.lower()).strip()
        phrase_text = non_word_re.sub(" ", phrase_text.lower()).strip()
        if not concept or not phrase_text:
            return False
        tokens = concept.split()
        if len(tokens) == 1:
            core = self._plural_pattern(tokens[0])
        else:
            core = r"\s+".join(re.escape(t) for t in tokens[:-1])
            core = core + r"\s+" + self._plural_pattern(tokens[-1])
        pattern = r"\b" + core + r"\b"
        return re.search(pattern, phrase_text) is not None


    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


    def _parse_class_names(self, class_names: Sequence[str]) -> List[str]:
        return [self._normalize_name(name) for name in class_names]


    def _read_txt_image_list(self, txt_path: str) -> List[str]:
        items: List[str] = []
        if not os.path.exists(txt_path):
            return items
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(os.path.basename(line))
        return items


    def _read_json_image_list_and_bbox(self, json_path: str) -> Tuple[List[str], Dict[str, List[float]]]:
        items: List[str] = []
        bbox_map: Dict[str, List[float]] = {}
        if not os.path.exists(json_path):
            return items, bbox_map

        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for row in payload.get("images", []):
            file_name = row.get("file_name")
            if not file_name:
                img_id = row.get("img_id")
                if img_id is None:
                    continue
                file_name = f"{img_id}.jpg"
            file_name = os.path.basename(str(file_name))
            bbox = row.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            bbox = [float(v) for v in bbox]
            items.append(file_name)
            # Allow duplicated samples in json (used for per-concept sample count padding).
            # Keep a single bbox per file for crop lookup.
            if file_name not in bbox_map:
                bbox_map[file_name] = bbox
        return items, bbox_map


    def _unique_keep_order(self, items: Sequence[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out


    def _load_images_from_df_lists(self, 
        df_root: str,
        split: str,
        item_folder: str,
        class_names: Sequence[str],
    ) -> List[str]:
        images: List[str] = []
        for class_name in class_names:
            txt_path = os.path.join(df_root, split, "Df", item_folder, f"{self._normalize_name(class_name)}.txt")
            images.extend(self._read_txt_image_list(txt_path))
        return self._unique_keep_order(images)


    def _load_images_and_bbox_from_df_lists(
        self,
        df_root: str,
        split: str,
        item_folder: str,
        class_names: Sequence[str],
        list_format: str = "txt",
    ) -> Tuple[List[str], Dict[str, List[float]]]:
        images: List[str] = []
        bbox_map: Dict[str, List[float]] = {}
        if list_format == "txt":
            return self._load_images_from_df_lists(df_root, split, item_folder, class_names), bbox_map
        if list_format != "json":
            raise ValueError(f"Unsupported list format: {list_format}")

        for class_name in class_names:
            json_path = os.path.join(df_root, split, "Df", item_folder, f"{self._normalize_name(class_name)}.json")
            class_images, class_bbox_map = self._read_json_image_list_and_bbox(json_path)
            images.extend(class_images)
            for file_name, bbox in class_bbox_map.items():
                if file_name not in bbox_map:
                    bbox_map[file_name] = bbox
        return self._unique_keep_order(images), bbox_map


    def _build_forget_mask(self, forget_class_names: Sequence[str]) -> torch.Tensor:
        mask = torch.zeros(len(self.label_names), dtype=torch.float32)
        name_to_idx = {self._normalize_name(v): k for k, v in self.label_names.items()}
        for name in self._parse_class_names(forget_class_names):
            if name not in name_to_idx:
                raise ValueError(f"Unknown forget class name: {name}")
            mask[name_to_idx[name]] = 1.0
        return mask


    def _class_name_to_idx_map(self) -> Dict[str, int]:
        return {self._normalize_name(v): k for k, v in sorted(self.label_names.items())}


    def _load_test_samples_from_df_lists(
        self,
        df_root: str,
        split: str,
        item_folder: str,
        class_names: Sequence[str],
        list_format: str = "txt",
        max_per_class: int = 100,
    ) -> List[Dict[str, object]]:
        samples: List[Dict[str, object]] = []
        name_to_idx = self._class_name_to_idx_map()

        for class_name in class_names:
            norm_class = self._normalize_name(class_name)
            if norm_class not in name_to_idx:
                continue
            class_idx = int(name_to_idx[norm_class])

            if list_format == "txt":
                txt_path = os.path.join(df_root, split, "Df", item_folder, f"{norm_class}.txt")
                class_files = self._read_txt_image_list(txt_path)
                class_files = self._unique_keep_order(class_files)[: max(0, int(max_per_class))]
                for file_name in class_files:
                    samples.append(
                        {
                            "file_name": os.path.basename(file_name),
                            "eval_class_idx": class_idx,
                            "bbox": None,
                        }
                    )
                continue

            if list_format == "json":
                json_path = os.path.join(df_root, split, "Df", item_folder, f"{norm_class}.json")
                class_files, class_bbox_map = self._read_json_image_list_and_bbox(json_path)
                class_files = class_files[: max(0, int(max_per_class))]
                for file_name in class_files:
                    samples.append(
                        {
                            "file_name": os.path.basename(file_name),
                            "eval_class_idx": class_idx,
                            "bbox": class_bbox_map.get(file_name),
                        }
                    )
                continue

            raise ValueError(f"Unsupported list format: {list_format}")

        return samples


class COCODataSet(Dataset):
    """
    COCO multi-label dataset.
    Label rule:
    - Classes annotated in the image => 1
    - All other classes in 80-class space => 0
    """

    def __init__(
        self,
        label_names: Dict[int, str],
        annotation_file: Optional[str] = None,
        image_root: Optional[str] = None,
        coco_root: Optional[str] = None,
        split: str = "train",
        transform=None,
        return_meta: bool = False,
        filter_missing_images: bool = True,
        selected_files: Optional[Sequence[str]] = None,
        selected_samples: Optional[Sequence[Dict[str, object]]] = None,
        selected_bboxes: Optional[Dict[str, Sequence[float]]] = None,
        apply_bbox_crop: bool = False,
        forget_class_names: Optional[Sequence[str]] = None,
    ) -> None:
        if annotation_file is None:
            if coco_root is None:
                raise ValueError("Provide `annotation_file` or `coco_root`.")
            annotation_file = os.path.join(coco_root, "annotations", f"instances_{split}2017.json")
        if image_root is None:
            if coco_root is None:
                raise ValueError("Provide `image_root` or `coco_root`.")
            image_root = os.path.join(coco_root, f"{split}2017")

        self.label_names = label_names
        self.annotation_file = annotation_file
        self.image_root = image_root
        self.transform = transform
        self.return_meta = return_meta
        self.apply_bbox_crop = apply_bbox_crop
        self.num_classes = len(self.label_names)
        forget_class_names = forget_class_names or []

        self.unlearn_utils = UnlearnUtils(label_names)
        self.forget_mask = self.unlearn_utils._build_forget_mask(forget_class_names) if forget_class_names else torch.zeros(self.num_classes)

        self.coco = COCO(self.annotation_file)

        # LABEL_NAMES index space (0..79) -> class name (underscore format)
        self.idx_to_name: Dict[int, str] = dict(sorted(self.label_names.items()))
        self.name_to_idx: Dict[str, int] = {
            self.unlearn_utils._normalize_name(name): idx for idx, name in self.idx_to_name.items()
        }

        # COCO category id -> class index in LABEL_NAMES
        self.cat_id_to_class_idx: Dict[int, int] = {}
        for cat in self.coco.loadCats(self.coco.getCatIds()):
            norm_name = self.unlearn_utils._normalize_name(cat["name"])
            if norm_name in self.name_to_idx:
                self.cat_id_to_class_idx[cat["id"]] = self.name_to_idx[norm_name]

        self.image_ids: List[int] = []
        self.file_names: List[str] = []
        self.targets: List[np.ndarray] = []
        self.eval_class_indices: List[int] = []
        self.sample_bboxes: List[Optional[List[float]]] = []
        self.file_name_to_bbox: Dict[str, List[float]] = {
            os.path.basename(k): [float(v) for v in vals]
            for k, vals in (selected_bboxes or {}).items()
            if vals is not None and len(vals) == 4
        }
        target_cache: Dict[int, np.ndarray] = {}

        def _build_target(img_id: int) -> np.ndarray:
            cached = target_cache.get(img_id)
            if cached is not None:
                return cached
            target = np.zeros(self.num_classes, dtype=np.float32)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                class_idx = self.cat_id_to_class_idx.get(ann["category_id"])
                if class_idx is not None:
                    target[class_idx] = 1.0
            target_cache[img_id] = target
            return target

        if selected_samples is not None:
            file_to_img_id: Dict[str, int] = {}
            for img_id in self.coco.getImgIds():
                img_info = self.coco.loadImgs([img_id])[0]
                file_to_img_id[img_info["file_name"]] = img_id

            for sample in selected_samples:
                file_name = os.path.basename(str(sample.get("file_name", "")))
                if not file_name:
                    continue
                img_id = file_to_img_id.get(file_name)
                if img_id is None:
                    continue
                image_path = os.path.join(self.image_root, file_name)
                if filter_missing_images and (not os.path.exists(image_path)):
                    continue

                bbox = sample.get("bbox")
                bbox_out = None
                if bbox is not None and len(bbox) == 4:
                    bbox_out = [float(v) for v in bbox]
                eval_idx = int(sample.get("eval_class_idx", -1))

                self.image_ids.append(img_id)
                self.file_names.append(file_name)
                self.targets.append(_build_target(img_id).copy())
                self.eval_class_indices.append(eval_idx)
                self.sample_bboxes.append(bbox_out)
        else:
            selected_set = None
            if selected_files is not None:
                selected_set = {os.path.basename(x) for x in selected_files}

            for img_id in self.coco.getImgIds():
                img_info = self.coco.loadImgs([img_id])[0]
                file_name = img_info["file_name"]
                if selected_set is not None and file_name not in selected_set:
                    continue
                image_path = os.path.join(self.image_root, file_name)
                if filter_missing_images and (not os.path.exists(image_path)):
                    continue

                self.image_ids.append(img_id)
                self.file_names.append(file_name)
                self.targets.append(_build_target(img_id).copy())
                self.eval_class_indices.append(-1)
                self.sample_bboxes.append(None)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        file_name = self.file_names[index]
        image_path = os.path.join(self.image_root, file_name)

        image = Image.open(image_path).convert("RGB")
        bbox = self.sample_bboxes[index]
        if bbox is None:
            bbox = self.file_name_to_bbox.get(file_name)
        if self.apply_bbox_crop and bbox is not None:
            x, y, w, h = [float(v) for v in bbox]
            x1 = max(0, int(np.floor(x)))
            y1 = max(0, int(np.floor(y)))
            x2 = min(image.width, int(np.ceil(x + w)))
            y2 = min(image.height, int(np.ceil(y + h)))
            if x2 > x1 and y2 > y1:
                image = image.crop((x1, y1, x2, y2))
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.unlearn_utils._pil_to_tensor(image)

        full_label = torch.from_numpy(self.targets[index].copy())
        forget_mask = self.forget_mask
        forget_label = full_label * forget_mask
        retain_label = full_label * (1.0 - forget_mask)
        bbox_tensor = torch.tensor(
            bbox if bbox is not None else [0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
        has_bbox = torch.tensor(bbox is not None, dtype=torch.bool)
        eval_class_idx = torch.tensor(self.eval_class_indices[index], dtype=torch.long)

        if self.return_meta:
            return {
                "image": image,
                "label": full_label,
                "forget_label": forget_label,
                "retain_label": retain_label,
                "bbox": bbox_tensor,
                "has_bbox": has_bbox,
                "eval_class_idx": eval_class_idx,
                "image_id": image_id,
                "file_name": file_name,
                "image_path": image_path,
            }
        return {
            "image": image,
            "image_path": image_path,
            "label": full_label,
            "forget_label": forget_label,
            "retain_label": retain_label,
            "bbox": bbox_tensor,
            "has_bbox": has_bbox,
            "eval_class_idx": eval_class_idx,
        }


class FlickrDataSet(Dataset):
    """
    Flickr30k Entities multi-label dataset built from phrase-level instances.
    """

    def __init__(
        self,
        label_names: Dict[int, str],
        annotation_file: Optional[str] = None,
        image_root: Optional[str] = None,
        split: str = "train",
        transform=None,
        return_meta: bool = False,
        filter_missing_images: bool = True,
        selected_files: Optional[Sequence[str]] = None,
        selected_samples: Optional[Sequence[Dict[str, object]]] = None,
        selected_bboxes: Optional[Dict[str, Sequence[float]]] = None,
        apply_bbox_crop: bool = False,
        forget_class_names: Optional[Sequence[str]] = None,
    ) -> None:
        if annotation_file is None:
            raise ValueError("FlickrDataSet requires `annotation_file` (instances.json path).")
        if image_root is None:
            raise ValueError("FlickrDataSet requires `image_root`.")

        self.label_names = label_names
        self.annotation_file = annotation_file
        self.image_root = image_root
        self.transform = transform
        self.return_meta = return_meta
        self.apply_bbox_crop = apply_bbox_crop
        self.num_classes = len(self.label_names)
        forget_class_names = forget_class_names or []

        self.unlearn_utils = UnlearnUtils(label_names)
        self.forget_mask = self.unlearn_utils._build_forget_mask(forget_class_names) if forget_class_names else torch.zeros(self.num_classes)

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            instances_payload = json.load(f)

        self.idx_to_name: Dict[int, str] = dict(sorted(self.label_names.items()))
        self.name_to_idx: Dict[str, int] = {
            self.unlearn_utils._normalize_name(name): idx for idx, name in self.idx_to_name.items()
        }

        target_cache: Dict[str, np.ndarray] = {}

        def _build_target_from_instances(file_stem: str) -> np.ndarray:
            cached = target_cache.get(file_stem)
            if cached is not None:
                return cached
            target = np.zeros(self.num_classes, dtype=np.float32)
            image_instances = instances_payload.get(file_stem, {})
            for _, ins in image_instances.items():
                phrase = str(ins.get("instance", "")).strip().lower()
                if not phrase:
                    continue
                for class_idx, class_name in self.idx_to_name.items():
                    if self.unlearn_utils._concept_in_phrase(class_name.replace("_", " "), phrase):
                        target[int(class_idx)] = 1.0
            target_cache[file_stem] = target
            return target

        self.file_names: List[str] = []
        self.targets: List[np.ndarray] = []
        self.eval_class_indices: List[int] = []
        self.sample_bboxes: List[Optional[List[float]]] = []
        self.file_name_to_bbox: Dict[str, List[float]] = {
            os.path.basename(k): [float(v) for v in vals]
            for k, vals in (selected_bboxes or {}).items()
            if vals is not None and len(vals) == 4
        }

        if selected_samples is not None:
            for sample in selected_samples:
                file_name = os.path.basename(str(sample.get("file_name", "")))
                if not file_name:
                    continue
                file_stem = os.path.splitext(file_name)[0]
                if file_stem not in instances_payload:
                    continue
                image_path = os.path.join(self.image_root, file_name)
                if filter_missing_images and (not os.path.exists(image_path)):
                    continue

                bbox = sample.get("bbox")
                bbox_out = None
                if bbox is not None and len(bbox) == 4:
                    bbox_out = [float(v) for v in bbox]
                eval_idx = int(sample.get("eval_class_idx", -1))

                self.file_names.append(file_name)
                self.targets.append(_build_target_from_instances(file_stem).copy())
                self.eval_class_indices.append(eval_idx)
                self.sample_bboxes.append(bbox_out)
        else:
            selected_set = None
            if selected_files is not None:
                selected_set = {os.path.basename(x) for x in selected_files}

            for file_stem in instances_payload.keys():
                file_name = f"{file_stem}.jpg"
                if selected_set is not None and file_name not in selected_set:
                    continue
                image_path = os.path.join(self.image_root, file_name)
                if filter_missing_images and (not os.path.exists(image_path)):
                    continue
                self.file_names.append(file_name)
                self.targets.append(_build_target_from_instances(file_stem).copy())
                self.eval_class_indices.append(-1)
                self.sample_bboxes.append(None)

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int):
        file_name = self.file_names[index]
        image_path = os.path.join(self.image_root, file_name)

        image = Image.open(image_path).convert("RGB")
        bbox = self.sample_bboxes[index]
        if bbox is None:
            bbox = self.file_name_to_bbox.get(file_name)
        if self.apply_bbox_crop and bbox is not None:
            x, y, w, h = [float(v) for v in bbox]
            x1 = max(0, int(np.floor(x)))
            y1 = max(0, int(np.floor(y)))
            x2 = min(image.width, int(np.ceil(x + w)))
            y2 = min(image.height, int(np.ceil(y + h)))
            if x2 > x1 and y2 > y1:
                image = image.crop((x1, y1, x2, y2))
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.unlearn_utils._pil_to_tensor(image)

        full_label = torch.from_numpy(self.targets[index].copy())
        forget_mask = self.forget_mask
        forget_label = full_label * forget_mask
        retain_label = full_label * (1.0 - forget_mask)
        bbox_tensor = torch.tensor(
            bbox if bbox is not None else [0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
        has_bbox = torch.tensor(bbox is not None, dtype=torch.bool)
        eval_class_idx = torch.tensor(self.eval_class_indices[index], dtype=torch.long)

        if self.return_meta:
            return {
                "image": image,
                "label": full_label,
                "forget_label": forget_label,
                "retain_label": retain_label,
                "bbox": bbox_tensor,
                "has_bbox": has_bbox,
                "eval_class_idx": eval_class_idx,
                "image_id": os.path.splitext(file_name)[0],
                "file_name": file_name,
                "image_path": image_path,
            }
        return {
            "image": image,
            "image_path": image_path,
            "label": full_label,
            "forget_label": forget_label,
            "retain_label": retain_label,
            "bbox": bbox_tensor,
            "has_bbox": has_bbox,
            "eval_class_idx": eval_class_idx,
        }


def get_finegrained_dataset_cls(dataset_name: str):
    normalized = str(dataset_name).strip().lower()
    if normalized in {"coco", "coco2017", "coco2017_instances"}:
        return COCODataSet
    if normalized in {"flickr", "flickr30k", "flickr30k_entities"}:
        return FlickrDataSet
    raise ValueError(f"Unsupported dataset: {dataset_name}")


# class COCODataLoader(DataLoader):
#     def __init__(
#         self,
#         annotation_file: Optional[str] = None,
#         image_root: Optional[str] = None,
#         coco_root: Optional[str] = None,
#         split: str = "train",
#         transform=None,
#         return_meta: bool = False,
#         filter_missing_images: bool = True,
#         selected_files: Optional[Sequence[str]] = None,
#         selected_samples: Optional[Sequence[Dict[str, object]]] = None,
#         selected_bboxes: Optional[Dict[str, Sequence[float]]] = None,
#         apply_bbox_crop: bool = False,
#         forget_class_names: Optional[Sequence[str]] = None,
#         batch_size: int = 32,
#         shuffle: bool = True,
#         num_workers: int = 4,
#         pin_memory: bool = True,
#         drop_last: bool = False,
#     ) -> None:
#         dataset = COCODataSet(
#             annotation_file=annotation_file,
#             image_root=image_root,
#             coco_root=coco_root,
#             split=split,
#             transform=transform,
#             return_meta=return_meta,
#             filter_missing_images=filter_missing_images,
#             selected_files=selected_files,
#             selected_samples=selected_samples,
#             selected_bboxes=selected_bboxes,
#             apply_bbox_crop=apply_bbox_crop,
#             forget_class_names=forget_class_names,
#         )
#         super().__init__(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#             drop_last=drop_last,
#         )


# def build_train_dataset(args, transform=None) -> COCODataSet:
#     forget_class_names = _parse_class_names(args.forget_classes)
#     selected_files = _load_images_from_df_lists(
#         df_root=args.df_root,
#         split=args.train_split,
#         item_folder=args.train_item_folder,
#         class_names=forget_class_names,
#     )
#     return COCODataSet(
#         annotation_file=args.train_annotation_file,
#         image_root=args.train_image_root,
#         split=args.train_split,
#         transform=transform,
#         return_meta=args.return_meta,
#         selected_files=selected_files,
#         forget_class_names=forget_class_names,
#     )


# def build_train_dataloader(args, transform=None) -> DataLoader:
#     dataset = build_train_dataset(args, transform=transform)
#     return DataLoader(
#         dataset=dataset,
#         batch_size=args.batch_size,
#         shuffle=args.shuffle_train,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_memory,
#         drop_last=args.drop_last,
#     )


# def build_test_datasets(args, transform=None) -> Tuple[COCODataSet, COCODataSet]:
#     forget_class_names = _parse_class_names(args.forget_classes)
#     all_class_names = [_normalize_name(v) for _, v in sorted(LABEL_NAMES.items())]
#     forget_set = set(forget_class_names)
#     retain_class_names = [c for c in all_class_names if c not in forget_set]

#     forget_test_samples = _load_test_samples_from_df_lists(
#         df_root=args.df_root,
#         split=args.val_split,
#         item_folder=args.test_item_folder,
#         class_names=forget_class_names,
#         list_format=args.test_item_format,
#         max_per_class=args.test_max_per_class,
#     )
#     retain_test_samples = _load_test_samples_from_df_lists(
#         df_root=args.df_root,
#         split=args.val_split,
#         item_folder=args.test_item_folder,
#         class_names=retain_class_names,
#         list_format=args.test_item_format,
#         max_per_class=args.test_max_per_class,
#     )
#     apply_bbox_crop = args.test_item_format == "json"

#     forget_dataset = COCODataSet(
#         annotation_file=args.val_annotation_file,
#         image_root=args.val_image_root,
#         split=args.val_split,
#         transform=transform,
#         return_meta=args.return_meta,
#         selected_samples=forget_test_samples,
#         apply_bbox_crop=apply_bbox_crop,
#         forget_class_names=forget_class_names,
#     )
#     retain_dataset = COCODataSet(
#         annotation_file=args.val_annotation_file,
#         image_root=args.val_image_root,
#         split=args.val_split,
#         transform=transform,
#         return_meta=args.return_meta,
#         selected_samples=retain_test_samples,
#         apply_bbox_crop=apply_bbox_crop,
#         forget_class_names=forget_class_names,
#     )
#     return forget_dataset, retain_dataset


# def build_test_dataloaders(args, transform=None) -> Tuple[DataLoader, DataLoader]:
#     forget_dataset, retain_dataset = build_test_datasets(args, transform=transform)
#     forget_loader = DataLoader(
#         dataset=forget_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_memory,
#         drop_last=False,
#     )
#     retain_loader = DataLoader(
#         dataset=retain_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_memory,
#         drop_last=False,
#     )
#     return forget_loader, retain_loader


# def build_all_dataloaders(args, transform=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
#     train_loader = build_train_dataloader(args, transform=transform)
#     forget_test_loader, retain_test_loader = build_test_dataloaders(args, transform=transform)
#     return train_loader, forget_test_loader, retain_test_loader
