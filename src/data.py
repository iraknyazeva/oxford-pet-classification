import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

SEED = 42

# фиксируем seed
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    обертка над torch.utils.data.Subset., 
    чтобы применить разные трансформации к train и val/test
    '''
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)

    '''
    трансформации для train и val/test
    
    train: случайные аугментации + нормализация imagenet
    val/test: resize + center crop + нормализация
    '''
def get_transforms():
    # cреднее и стандартное отклонение imagenet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # трансформации для train
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),         
        transforms.RandomHorizontalFlip(),         
        transforms.ColorJitter(                    
            brightness=0.2, contrast=0.2, saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # трансформации для val/test
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, val_transform


def get_dataloaders(root='./data', batch_size=32, val_ratio=0.2, num_workers=0, seed=SEED):

    set_seed(seed)

    # создаем папку для данных, если ее нет 
    root = Path(root)
    root.mkdir(exist_ok=True)

    full_dataset = datasets.OxfordIIITPet(
        root=root,
        split='trainval',
        target_types='category',
        download=True,
        transform=None,         
    )

    num_classes = len(full_dataset.classes)

    # размеры train и val
    train_size = int((1 - val_ratio) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # случайное разбиение
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # получаем трансформации и оборачиваем subset
    train_tf, val_tf = get_transforms()
    train_ds = SubsetWithTransform(train_subset, train_tf)
    val_ds = SubsetWithTransform(val_subset, val_tf)

    # отдельный test набор
    test_dataset = datasets.OxfordIIITPet(
        root=root,
        split='test',
        target_types='category',
        download=False,        
        transform=val_tf,
    )

    # DataLoader'ы
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,            # перемешиваем только train
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_classes, full_dataset.classes