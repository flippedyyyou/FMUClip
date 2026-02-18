import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from finegrained.coco_labels.coco import LABEL_NAMES


def _normalize_name(name: str) -> str:
    return name.strip().replace(" ", "_")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _parse_class_names(class_names: Sequence[str]) -> List[str]:
    return [_normalize_name(name) for name in class_names]


def _read_txt_image_list(txt_path: str) -> List[str]:
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


def _unique_keep_order(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _load_images_from_df_lists(
    df_root: str,
    split: str,
    item_folder: str,
    class_names: Sequence[str],
) -> List[str]:
    images: List[str] = []
    for class_name in class_names:
        txt_path = os.path.join(df_root, split, "Df", item_folder, f"{_normalize_name(class_name)}.txt")
        images.extend(_read_txt_image_list(txt_path))
    return _unique_keep_order(images)


def _build_forget_mask(forget_class_names: Sequence[str]) -> torch.Tensor:
    mask = torch.zeros(len(LABEL_NAMES), dtype=torch.float32)
    name_to_idx = {_normalize_name(v): k for k, v in LABEL_NAMES.items()}
    for name in _parse_class_names(forget_class_names):
        if name not in name_to_idx:
            raise ValueError(f"Unknown forget class name: {name}")
        mask[name_to_idx[name]] = 1.0
    return mask


class COCODataSet(Dataset):
    """
    COCO multi-label dataset.
    Label rule:
    - Classes annotated in the image => 1
    - All other classes in 80-class space => 0
    """

    def __init__(
        self,
        annotation_file: Optional[str] = None,
        image_root: Optional[str] = None,
        coco_root: Optional[str] = None,
        split: str = "train",
        transform=None,
        return_meta: bool = False,
        filter_missing_images: bool = True,
        selected_files: Optional[Sequence[str]] = None,
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

        self.annotation_file = annotation_file
        self.image_root = image_root
        self.transform = transform
        self.return_meta = return_meta
        self.num_classes = len(LABEL_NAMES)
        forget_class_names = forget_class_names or []
        self.forget_mask = _build_forget_mask(forget_class_names) if forget_class_names else torch.zeros(self.num_classes)

        self.coco = COCO(self.annotation_file)

        # LABEL_NAMES index space (0..79) -> class name (underscore format)
        self.idx_to_name: Dict[int, str] = dict(sorted(LABEL_NAMES.items()))
        self.name_to_idx: Dict[str, int] = {
            _normalize_name(name): idx for idx, name in self.idx_to_name.items()
        }

        # COCO category id -> class index in LABEL_NAMES
        self.cat_id_to_class_idx: Dict[int, int] = {}
        for cat in self.coco.loadCats(self.coco.getCatIds()):
            norm_name = _normalize_name(cat["name"])
            if norm_name in self.name_to_idx:
                self.cat_id_to_class_idx[cat["id"]] = self.name_to_idx[norm_name]

        self.image_ids: List[int] = []
        self.file_names: List[str] = []
        self.targets: List[np.ndarray] = []
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

            target = np.zeros(self.num_classes, dtype=np.float32)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                class_idx = self.cat_id_to_class_idx.get(ann["category_id"])
                if class_idx is not None:
                    target[class_idx] = 1.0

            self.image_ids.append(img_id)
            self.file_names.append(file_name)
            self.targets.append(target)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        file_name = self.file_names[index]
        image_path = os.path.join(self.image_root, file_name)

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = _pil_to_tensor(image)

        full_label = torch.from_numpy(self.targets[index].copy())
        forget_mask = self.forget_mask
        forget_label = full_label * forget_mask
        retain_label = full_label * (1.0 - forget_mask)

        if self.return_meta:
            return {
                "image": image,
                "label": full_label,
                "forget_label": forget_label,
                "retain_label": retain_label,
                "image_id": image_id,
                "file_name": file_name,
                "image_path": image_path,
            }
        return {
            "image": image,
            "label": full_label,
            "forget_label": forget_label,
            "retain_label": retain_label,
        }


class COCODataLoader(DataLoader):
    def __init__(
        self,
        annotation_file: Optional[str] = None,
        image_root: Optional[str] = None,
        coco_root: Optional[str] = None,
        split: str = "train",
        transform=None,
        return_meta: bool = False,
        filter_missing_images: bool = True,
        selected_files: Optional[Sequence[str]] = None,
        forget_class_names: Optional[Sequence[str]] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> None:
        dataset = COCODataSet(
            annotation_file=annotation_file,
            image_root=image_root,
            coco_root=coco_root,
            split=split,
            transform=transform,
            return_meta=return_meta,
            filter_missing_images=filter_missing_images,
            selected_files=selected_files,
            forget_class_names=forget_class_names,
        )
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )


def build_train_dataset(args, transform=None) -> COCODataSet:
    forget_class_names = _parse_class_names(args.forget_classes)
    selected_files = _load_images_from_df_lists(
        df_root=args.df_root,
        split=args.train_split,
        item_folder=args.train_item_folder,
        class_names=forget_class_names,
    )
    return COCODataSet(
        annotation_file=args.train_annotation_file,
        image_root=args.train_image_root,
        split=args.train_split,
        transform=transform,
        return_meta=args.return_meta,
        selected_files=selected_files,
        forget_class_names=forget_class_names,
    )


def build_train_dataloader(args, transform=None) -> DataLoader:
    dataset = build_train_dataset(args, transform=transform)
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )


def build_test_datasets(args, transform=None) -> Tuple[COCODataSet, COCODataSet]:
    forget_class_names = _parse_class_names(args.forget_classes)
    all_class_names = [_normalize_name(v) for _, v in sorted(LABEL_NAMES.items())]
    forget_set = set(forget_class_names)
    retain_class_names = [c for c in all_class_names if c not in forget_set]

    forget_test_files = _load_images_from_df_lists(
        df_root=args.df_root,
        split=args.val_split,
        item_folder=args.test_item_folder,
        class_names=forget_class_names,
    )
    retain_test_files = _load_images_from_df_lists(
        df_root=args.df_root,
        split=args.val_split,
        item_folder=args.test_item_folder,
        class_names=retain_class_names,
    )

    forget_dataset = COCODataSet(
        annotation_file=args.val_annotation_file,
        image_root=args.val_image_root,
        split=args.val_split,
        transform=transform,
        return_meta=args.return_meta,
        selected_files=forget_test_files,
        forget_class_names=forget_class_names,
    )
    retain_dataset = COCODataSet(
        annotation_file=args.val_annotation_file,
        image_root=args.val_image_root,
        split=args.val_split,
        transform=transform,
        return_meta=args.return_meta,
        selected_files=retain_test_files,
        forget_class_names=forget_class_names,
    )
    return forget_dataset, retain_dataset


def build_test_dataloaders(args, transform=None) -> Tuple[DataLoader, DataLoader]:
    forget_dataset, retain_dataset = build_test_datasets(args, transform=transform)
    forget_loader = DataLoader(
        dataset=forget_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    retain_loader = DataLoader(
        dataset=retain_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    return forget_loader, retain_loader


def build_all_dataloaders(args, transform=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = build_train_dataloader(args, transform=transform)
    forget_test_loader, retain_test_loader = build_test_dataloaders(args, transform=transform)
    return train_loader, forget_test_loader, retain_test_loader
