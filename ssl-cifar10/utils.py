# Utility functions for data augmentation and other helper functions.

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import AutoAugmentedDataset, WeakStrongAugmentDataset

def RandMagAugment(num_ops, magnitude_max, num_magnitude_bins):
    rand_mag = np.random.randint(1, magnitude_max)
    return transforms.RandAugment(num_ops=num_ops, magnitude=rand_mag, num_magnitude_bins=num_magnitude_bins)

def create_data_loaders(trainset, testset, batch_size, ratio_unlabeled_labeled, seed):
    # Set the random seeds for reproducible behavior
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define weak and strong augmentations
    weak_transform = transforms.Compose([
        transforms.RandomHorizontalFlip()
    ])
    strong_transform = transforms.Compose([
        # Pytorch RandAugment does approximately the same as in the fixmatch paper  
        # https://pytorch.org/vision/main/_modules/torchvision/transforms/autoaugment.html#RandAugment
        RandMagAugment(num_ops=2, magnitude_max = 10, num_magnitude_bins= 31)
    ])

    # Select 250 random annotated images
    indices = np.random.choice(len(trainset), 250, replace=False)
    auto_augmented_data = AutoAugmentedDataset(trainset.data, trainset.targets, trainset.classes)
    annotated_data = torch.utils.data.Subset(auto_augmented_data, indices)

    # Exclude these 250 images from the non-annotated dataset
    mask = np.ones(len(trainset), dtype=bool)
    mask[indices] = False
    weak_strong_augmented_data = WeakStrongAugmentDataset(trainset.data, trainset.classes, weak_transform, strong_transform)
    non_annotated_data = torch.utils.data.Subset(weak_strong_augmented_data, np.arange(len(weak_strong_augmented_data))[mask])
    
    # Create data loaders
    annotated_loader = DataLoader(annotated_data, batch_size=batch_size, shuffle=True, num_workers=2)
    non_annotated_loader = DataLoader(non_annotated_data, batch_size=ratio_unlabeled_labeled*batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return annotated_loader, non_annotated_loader, test_loader