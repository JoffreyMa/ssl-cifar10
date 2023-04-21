# Main script to run the training and evaluation process.

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from wide_resnet import WideResNet
from fixmatch import FixMatch
from utils import create_data_loaders
from evaluate import evaluate
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

def download_cifar10(path='./data'):
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(os.path.join(path, 'cifar-10-batches-py')):
        _ = torchvision.datasets.CIFAR10(root=path, train=True, download=True)
        _ = torchvision.datasets.CIFAR10(root=path, train=False, download=True)


def main():
    # Parameters
    # Optimizer
    lr = 0.03
    momentum = 0.9
    weight_decay = 5e-4
    # Training
    nb_epochs = 1000
    # Dataloader
    batch_size=50
    ratio_unlabeled_labeled=7
    seed=42
    # Model
    depth=28 
    widen_factor=2
    dropout_rate=0.3
    num_classes=10

    # Set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download data if necessary
    download_cifar10()
    
    # Instanciate train and test sets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
    
    # Instanciate data loaders for WideResNet with FixMatch
    labeled_loader, unlabeled_loader, test_loader = create_data_loaders(trainset, testset, batch_size=batch_size, ratio_unlabeled_labeled=ratio_unlabeled_labeled, seed=seed)
    
    # Declare WideResNet
    model = WideResNet(depth=depth, widen_factor=widen_factor, dropout_rate=dropout_rate, num_classes=num_classes)
    model.to(device)
    
    # Declare the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Create the learning rate scheduler
    # Differ from the fixmatch paper scheduler but is in the same spirit
    total_training_steps = len(labeled_loader) * nb_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps, eta_min=0, last_epoch=-1)
    
    # Declare the FixMatch
    fixmatch = FixMatch(model, device, optimizer, scheduler, labeled_loader, unlabeled_loader, test_loader)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ssl-cifar10",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": "WideResNet-28-2",
        "dataset": "CIFAR-10",
        "epochs": nb_epochs,
        }
    )
    
    for epoch in range(nb_epochs):
        fixmatch.train()
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)
        print(f"Epoch: {epoch}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")
        # log metrics to wandb
        wandb.log({"Test Loss": test_accuracy, "Test Accuracy": test_loss, "Test Precision" : test_precision, "Test Recall" : test_recall, "Test F1" : test_f1})
        
    torch.save(model.state_dict(), os.path.join("models", "fixmatch_wide_resnet.pth"))

if __name__ == "__main__":
    main()