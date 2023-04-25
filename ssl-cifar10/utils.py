# Utility functions for data augmentation and other helper functions.

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import AutoAugmentedDataset, WeakStrongAugmentDataset
import os
import torchvision


def download_cifar10(path='./data'):
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(os.path.join(path, 'cifar-10-batches-py')):
        _ = torchvision.datasets.CIFAR10(root=path, train=True, download=True)
        _ = torchvision.datasets.CIFAR10(root=path, train=False, download=True)


def save_transformed_images(image_tensors, output_dir, prefix):
    to_pil_image = transforms.ToPILImage()

    # Check if the output folder exists and create it if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through the image tensors in the batch
    for i, image_tensor in enumerate(image_tensors):
        # Convert the image tensor to a PIL image
        pil_transformed_image = to_pil_image(image_tensor.cpu())

        # Save the PIL image to the output directory with a unique name based on the prefix and index
        output_name = f"{prefix}_{i}.png"
        pil_transformed_image.save(os.path.join(output_dir, output_name))


def RandMagAugment(num_ops, magnitude_max, num_magnitude_bins):
    rand_mag = np.random.randint(1, magnitude_max)
    return transforms.RandAugment(num_ops=num_ops, magnitude=rand_mag, num_magnitude_bins=num_magnitude_bins)


def RandErasing(ratio_range):
    # Sightly different from the original paper
    # Erase either 5, 10, 15 percent of the image
    rand_scale = 0.05 * np.random.randint(low=1, high=4, size=1)[0]
    return transforms.RandomErasing(p=1, scale=(rand_scale, rand_scale), ratio=ratio_range)


def create_data_loaders(trainset, testset, batch_size, ratio_unlabeled_labeled, nb_steps, seed):
    # Set the random seeds for reproducible behavior
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define weak and strong augmentations
    # flip-and-shift data augmentation
    weak_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])
    # RandAugment & Erasing
    strong_transform = transforms.Compose([
        # Pytorch RandAugment does approximately the same as in the fixmatch paper  
        # https://pytorch.org/vision/main/_modules/torchvision/transforms/autoaugment.html#RandAugment
        RandMagAugment(num_ops=2, magnitude_max = 10, num_magnitude_bins= 10),
        transforms.PILToTensor(), # does not scale !
        transforms.ConvertImageDtype(torch.float32), # scales !
        RandErasing(ratio_range = (0.5, 5))
    ])

    # Select 250 random annotated images
    indices = np.random.choice(len(trainset), 250, replace=False)
    auto_augmented_data = AutoAugmentedDataset(trainset.data, trainset.targets, trainset.classes, nb_steps, batch_size)
    annotated_data = torch.utils.data.Subset(auto_augmented_data, indices)

    # Exclude these 250 images from the non-annotated dataset
    mask = np.ones(len(trainset), dtype=bool)
    mask[indices] = False
    weak_strong_augmented_data = WeakStrongAugmentDataset(trainset.data, trainset.classes, weak_transform, strong_transform, batch_size)
    non_annotated_data = torch.utils.data.Subset(weak_strong_augmented_data, np.arange(len(weak_strong_augmented_data))[mask])
    
    # Create data loaders
    annotated_loader = DataLoader(annotated_data, batch_size=batch_size, shuffle=True, num_workers=2)
    non_annotated_loader = DataLoader(non_annotated_data, batch_size=ratio_unlabeled_labeled*batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return annotated_loader, non_annotated_loader, test_loader