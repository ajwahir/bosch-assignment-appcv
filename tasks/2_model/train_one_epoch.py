"""Small training script: runs one epoch on a subset of BDD100K and saves a checkpoint.

Usage:
    python3 tasks/2_model/train_one_epoch.py --subset 256 --batch 4 --epochs 1
"""
import argparse
import random
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

from dataset import BDD100KDataset, build_class_map, collate_fn
from model_utils import build_model


def train(repo_root: Path, subset_size: int = 256, batch_size: int = 4, epochs: int = 1, device: str = None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    repo_root = Path(repo_root)
    ann_file = repo_root / 'data' / 'bdd100k_labels_release' / 'bdd100k' / 'labels' / 'bdd100k_labels_images_train.json'
    with open(ann_file, 'r') as f:
        annotations = json.load(f)

    images_dir = repo_root / 'data' / 'bdd100k_images_100k' / 'bdd100k' / 'images' / '100k' / 'train'

    class_map = build_class_map(annotations)

    ds = BDD100KDataset(annotations, str(images_dir), class_map, exclude_classes=['lane', 'drivable area'])
    # small random subset for fast iteration
    indices = list(range(len(ds)))
    random.seed(0)
    random.shuffle(indices)
    indices = indices[:subset_size]
    ds_small = Subset(ds, indices)

    loader = DataLoader(ds_small, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=1)

    model = build_model(num_classes=len(class_map) + 1, pretrained=False)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    model.train()
    for epoch in range(epochs):
        for images, targets in loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            print(f'Epoch {epoch} - loss: {losses.item():.4f}')
        print(f'Epoch {epoch} completed')

    ckpt = repo_root / 'checkpoints'
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt / 'fasterrcnn_one_epoch.pth')
    print('Saved checkpoint to', ckpt / 'fasterrcnn_one_epoch.pth')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo_root', default=Path(__file__).resolve().parents[2])
    p.add_argument('--subset', type=int, default=256)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--epochs', type=int, default=1)
    args = p.parse_args()
    train(Path(args.repo_root), subset_size=args.subset, batch_size=args.batch, epochs=args.epochs)


if __name__ == '__main__':
    main()
