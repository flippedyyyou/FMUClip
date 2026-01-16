import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from PIL import Image
import os
import pandas as pd
import torchvision
import json

from typing import *


__all__ = ["UniversalDataset"]


class CustomizedDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.forget_rule = lambda x: False

    def is_in_forget_set(self, sample) -> bool:
        return self.forget_rule(sample)

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: False) -> None:
        self.forget_rule = fn
        return


class CC12MDataset(CustomizedDataset):
    def __init__(self, annotations_file, img_dir, *args, **kwargs):
        super().__init__()
        with open(annotations_file, "r") as f:
            self.metadata = json.load(f)
        self.img_dir = img_dir

        self.backup_image_name = "00191154.jpg"
        self.backup_text = "Vector seamless pattern with flower Chinese plum. Floral pattern with leaves flowers and branches of the tree Chinese plum. Design paper wallpaper and fabrics. Black red gold."

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata[idx]["img_name"]
        text = self.metadata[idx]["text"]

        if isinstance(text, str):
            text = text.strip()
        else:
            text = ""

        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            if image.size[0] < 5 or image.size[1] < 5:
                img_path = os.path.join(self.img_dir, self.backup_image_name)
                image = Image.open(img_path).convert("RGB")
                text = self.backup_text
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {img_path}")

        sample = {
            "image": image,
            "text": text,
        }

        return sample


class Flickr30kDataset(CustomizedDataset):
    def __init__(self, annotations_file, img_dir, *args, **kwargs):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file, delimiter="|")
        self.img_dir = img_dir
        self.forget_set_image_names = set()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0].strip()
        caption = self.img_labels.iloc[idx, 2]  # Removed strip() for now

        # Check if the caption is a string and not NaN (float)
        if isinstance(caption, str):
            caption = caption.strip()
        else:
            caption = ""  # Or some placeholder text like "<missing caption>"

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        is_in_forget_set = img_name in self.forget_set_image_names

        return {"image": image, "text": caption, "is_in_forget_set": is_in_forget_set}

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: False) -> None:

        retain_set_image_names = set()

        for idx in range(len(self)):
            img_name = self.img_labels.iloc[idx, 0].strip()
            caption = self.img_labels.iloc[idx, 2]  # Removed strip() for now

            # Check if the caption is a string and not NaN (float)
            if isinstance(caption, str):
                caption = caption.strip()
            else:
                caption = ""  # Or some placeholder text like "<missing caption>"

            if fn(caption):
                self.forget_set_image_names.add(img_name)
            else:
                retain_set_image_names.add(img_name)

        print(
            f"Statistics: Forget set {len(self.forget_set_image_names)} images, Retain set {len(retain_set_image_names)}."
        )
        self.forget_rule = lambda sample: sample["is_in_forget_set"]
        return


class CIFAR100Dataset(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        self.label_names: Dict[str, Any] = kwargs.get("label_names", {})
        del kwargs["label_names"]
        super().__init__(*args, **kwargs)
        self.forget_rule = lambda x: False

    def is_in_forget_set(self, sample) -> bool:
        return self.forget_rule(sample)

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: False) -> None:
        self.forget_rule = fn

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        text = self.label_names[label]
        return {"image": image, "text": text, "label": label}


class UniversalDataset(Dataset):
    def __init__(self, dataset_name: str, *args: Any, **kwargs: Any):
        self.dataset_name_to_class: Dict[str, object] = {
            "Flickr30k": Flickr30kDataset,
            "CIFAR100": CIFAR100Dataset,
            "CC12M": CC12MDataset,
        }
        DatasetCls = self.dataset_name_to_class.get(dataset_name, None)
        if DatasetCls:
            self.dataset: Union[Dataset, List] = DatasetCls(*args, **kwargs)
        else:
            self.dataset = []
        return

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return {
            "image": self.dataset[index]["image"],
            "text": self.dataset[index]["text"],
            "flag": self.dataset.is_in_forget_set(self.dataset[index]),
        }

    def set_forget_rule(self, fn: Callable[[Any], bool] = lambda x: True) -> None:
        self.dataset.set_forget_rule(fn)
        return


def UniversalDataLoader(*args, **kwargs):
    def collate_fn(data: List[Tuple[Any, Any, Any]]):
        images: list = []
        texts: list = []
        flags: list = []

        for sample in data:
            image, text, flag = sample["image"], sample["text"], sample["flag"]
            images.append(image)
            texts.append(text)
            flags.append(flag)
        return images, texts, flags

    return DataLoader(*args, collate_fn=collate_fn, **kwargs)


if __name__ == "__main__":
    dataset = UniversalDataset(
        "CC12M",
        "/datanfs4/shenruoyan/ClipErase-ACL/datasets/cc12m_metadata_new.json", # <-- change to your own path
        "/datanfs4/shenruoyan/ClipErase-ACL/datasets/cc12m_images",            # <-- change to your own path
    )
    dataset.set_forget_rule(
        lambda sample: "woman" in sample["text"].lower()
        or "women" in sample["text"].lower()
    )
    print(len(dataset))
