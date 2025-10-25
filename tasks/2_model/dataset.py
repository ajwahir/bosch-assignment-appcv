"""BDD100K PyTorch Dataset loader

Provides BDD100KDataset which yields (image_tensor, target) tuples compatible
with torchvision detection models. Also includes helper utilities to build a
class->id mapping from the annotations and a simple collate_fn.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


def build_class_map(annotations: List[Dict[str, Any]], whitelist: Optional[List[str]] = None) -> Dict[str, int]:
    """Build a mapping from category name to integer id (starting at 1).

    If `whitelist` is provided, only categories in the whitelist are included.
    """
    cnt = {}
    for item in annotations:
        for lab in item.get('labels', []):
            cat = lab.get('category')
            if not cat:
                continue
            if whitelist and cat not in whitelist:
                continue
            cnt[cat] = cnt.get(cat, 0) + 1
    cats = sorted(cnt.keys())
    # ids start at 1 (0 reserved by some frameworks as background)
    return {c: i + 1 for i, c in enumerate(cats)}


class BDD100KDataset(Dataset):
    def __init__(self,
                 annotations: List[Dict[str, Any]],
                 images_dir: str,
                 class_map: Dict[str, int],
                 transforms=None,
                 exclude_classes: Optional[List[str]] = None):
        self.annotations = annotations
        self.images_dir = Path(images_dir)
        self.class_map = class_map
        self.transforms = transforms
        self.exclude = set(exclude_classes or [])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_name = item.get('name')
        img_path = self.images_dir / img_name
        # open image
        img = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for lab in item.get('labels', []):
            cat = lab.get('category')
            if not cat or cat in self.exclude:
                continue
            box = lab.get('box2d')
            if not box:
                continue
            x1, y1, x2, y2 = box.get('x1'), box.get('y1'), box.get('x2'), box.get('y2')
            if None in (x1, y1, x2, y2):
                continue
            # basic validity
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_map.get(cat, 0))
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(0)

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'iscrowd': iscrowd,
        }

        # apply transforms that expect PIL image + target (optional)
        if self.transforms:
            img, target = self.transforms(img, target)

        # ensure image is a Tensor
        if not isinstance(img, torch.Tensor):
            import torchvision.transforms as T
            img = T.ToTensor()(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
