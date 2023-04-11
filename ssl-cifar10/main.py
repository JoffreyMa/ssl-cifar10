# Main script to run the training and evaluation process.

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from wide_resnet import WideResNet
from fixmatch import FixMatch
from utils import create_data_loaders

def download_cifar10(path='./data'):
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(os.path.join(path, 'cifar-10-batches-py')):
        _ = torchvision.datasets.CIFAR10(root=path, train=True, download=True)
        _ = torchvision.datasets.CIFAR10(root=path, train=False, download=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    download_cifar10()
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
    
    labeled_loader, unlabeled_loader, test_loader = create_data_loaders(trainset, testset, batch_size=50, ratio_unlabeled_labeled=7, seed=42)
    
    model = WideResNet(depth=28, widen_factor=2, dropout_rate=0.3, num_classes=10)
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
    
    fixmatch = FixMatch(model, device, optimizer, labeled_loader, unlabeled_loader, test_loader)
    
    for epoch in range(1, 101):
        fixmatch.train()
        test_loss, test_acc = fixmatch.evaluate()
        print(f"Epoch: {epoch}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
    torch.save(model.state_dict(), "fixmatch_wide_resnet.pth")

if __name__ == "__main__":
    main()
