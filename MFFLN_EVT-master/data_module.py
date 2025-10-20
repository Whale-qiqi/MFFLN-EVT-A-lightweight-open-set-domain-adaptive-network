# data_module.py
# ------------------------------------------------------------
# Data loading utilities: dataset class, split by class, label
# inspection, GPU sync, and helper builders for datasets/
# dataloaders. All original functionality is preserved.
# ------------------------------------------------------------

from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import os
import torch
import numpy as np


class CustomDataset(Dataset):
    """Custom dataset scanning folders where each subdir = class name."""
    def __init__(self, root, transform, label_dict=None, unknown_label=None):
        self.data, self.targets = [], []
        self.transform = transform
        self.classes = sorted(os.listdir(root))

        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if not os.path.isfile(img_path):
                    continue
                if label_dict and unknown_label is not None:
                    label = label_dict.get(class_name, unknown_label)
                else:
                    label = self.classes.index(class_name)
                self.data.append(img_path)
                self.targets.append(label)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.targets[idx]


def split_by_class(dataset, train_ratio=0.9):
    """Split dataset into train/test by class."""
    targets = np.array(dataset.targets)
    train_idx, test_idx = [], []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        s = int(len(idx) * train_ratio)
        train_idx.extend(idx[:s]); test_idx.extend(idx[s:])
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def show_dataset_labels(dataset, subset, name):
    labels = [dataset.targets[i] for i in subset.indices]
    print(f"{name} Labels: {labels}")
    return labels


def gpu_sync():
    """Synchronize GPU (safe on CPU)."""
    if torch.cuda.is_available(): torch.cuda.synchronize()


def build_datasets(source_path, target_path, transform, label_dict, unknown_label):
    source_dataset = CustomDataset(root=source_path, transform=transform)
    target_dataset = CustomDataset(root=target_path, transform=transform,
                                   label_dict=label_dict, unknown_label=unknown_label)
    return source_dataset, target_dataset


def build_dataloaders(source_dataset, target_dataset, batch_size=64, train_ratio=0.9):
    s_train, s_test = split_by_class(source_dataset, train_ratio)
    t_train, t_test = split_by_class(target_dataset, train_ratio)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    loaders = {
        "source_loader": source_loader,
        "source_train_loader": DataLoader(s_train, batch_size=batch_size, shuffle=True),
        "source_test_loader": DataLoader(s_test, batch_size=batch_size, shuffle=False),
        "target_train_loader": DataLoader(t_train, batch_size=batch_size, shuffle=True),
        "target_test_loader": DataLoader(t_test, batch_size=batch_size, shuffle=False)
    }
    return loaders
