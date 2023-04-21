# Utility functions for data augmentation and other helper functions.

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os

def create_data_loaders(trainset, testset, batch_size, ratio_unlabeled_labeled, seed):
    # Set the random seeds for reproducible behavior
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create the data folder if it does not exist
    data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Check if transformed data already exists in the data folder
    annotated_data_path = os.path.join(data_path, "annotated_data.pt")
    non_annotated_data_path = os.path.join(data_path, "non_annotated_data.pt")
    
    if os.path.exists(annotated_data_path) and os.path.exists(non_annotated_data_path):
        # Load the transformed data
        annotated_data = torch.load(annotated_data_path)
        non_annotated_data = torch.load(non_annotated_data_path)
    else:
        # Select 250 random annotated images
        indices = np.random.choice(len(trainset), 250, replace=False)
        annotated_data = torch.utils.data.Subset(trainset, indices)

        # Exclude these 250 images from the non-annotated dataset
        mask = np.ones(len(trainset), dtype=bool)
        mask[indices] = False
        non_annotated_data = torch.utils.data.Subset(trainset, np.arange(len(trainset))[mask])
    
        # Apply data augmentation to non-annotated data and replace the labels
        unlabeled_weakly_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        unlabeled_strongly_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])

        unlabeled_weakly_transformed_data = []
        unlabeled_strongly_transformed_data = []
        for data, _ in non_annotated_data:
            unlabeled_weakly_transformed_data.append(unlabeled_weakly_transform(data))
            unlabeled_strongly_transformed_data.append(unlabeled_strongly_transform(data))
        unlabeled_weakly_transformed_data = torch.stack(unlabeled_weakly_transformed_data)
        unlabeled_strongly_transformed_data = torch.stack(unlabeled_strongly_transformed_data)
        non_annotated_data = torch.utils.data.TensorDataset(unlabeled_weakly_transformed_data, 
                                                            unlabeled_strongly_transformed_data)
        
        # Save data
        torch.save(annotated_data, annotated_data_path)
        torch.save(non_annotated_data, non_annotated_data_path)

    # Create data loaders
    annotated_loader = DataLoader(annotated_data, batch_size=batch_size, shuffle=True, num_workers=2)
    non_annotated_loader = DataLoader(non_annotated_data, batch_size=ratio_unlabeled_labeled*batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return annotated_loader, non_annotated_loader, test_loader
