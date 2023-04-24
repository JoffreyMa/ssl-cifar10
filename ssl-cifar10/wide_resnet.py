# Wide ResNet-28-2 model implementation.

import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import AutoAugmentedDataset, download_cifar10
from evaluate import evaluate
import wandb
import os
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout_rate, stride):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        
        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        assert (depth - 4) % 6 == 0, "Depth must be (6n+4)"
        n = (depth - 4) // 6

        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._wide_layer(BasicBlock, n_channels[0], n_channels[1], dropout_rate, n, stride=1)
        self.layer2 = self._wide_layer(BasicBlock, n_channels[1], n_channels[2], dropout_rate, n, stride=2)
        self.layer3 = self._wide_layer(BasicBlock, n_channels[2], n_channels[3], dropout_rate, n, stride=2)
        self.bn1 = nn.BatchNorm2d(n_channels[3], momentum=0.9)
        self.fc = nn.Linear(n_channels[3], num_classes)

        for m in self.modules():
            # Taken from https://github.com/meliketoy/wide-resnet.pytorch/blob/292b3ede0651e349dd566f9c23408aa572f1bd92/networks/wide_resnet.py
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def _wide_layer(self, block, in_planes, out_planes, dropout_rate, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(in_planes, out_planes, dropout_rate, stride))
            else:
                layers.append(block(out_planes, out_planes, dropout_rate, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()

    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = total_correct / len(dataloader.dataset)
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    # Try out WideResNet performances on fully labeled data
    # Parameters
    # Optimizer
    lr = 0.03
    momentum = 0.9
    weight_decay = 5e-4
    # Training
    nb_epochs = 100
    # Dataloader
    batch_size=64
    # Model
    depth=28 
    widen_factor=2
    dropout_rate=0.3
    # Goal
    num_classes=10
    # Log
    log_wandb = True

    # Set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download data if necessary
    download_cifar10()
    
    # Instanciate train and test sets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())

    # Instanciate data loaders
    auto_augmented_data = AutoAugmentedDataset(trainset.data, trainset.targets, trainset.classes)
    train_loader = DataLoader(auto_augmented_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Declare WideResNet
    model = WideResNet(depth=depth, widen_factor=widen_factor, dropout_rate=dropout_rate, num_classes=num_classes)
    model.to(device)
    
    # Declare the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # start a new wandb run to track this script
    if log_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="WideResNet-cifar10",
            
            # track hyperparameters and run metadata
            config={
            "architecture": "WideResNet-28-2",
            "dataset": "CIFAR-10",
            "epochs": nb_epochs,
            "learning_rate": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "depth": depth,
            "widen_factor": widen_factor,
            "dropout_rate": dropout_rate,
            }
        )
        
    # Training loop
    for epoch in range(nb_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        
        # Evaluate the model on the test set and print the results
        evaluation = evaluate({"Model":model}, {"Test":test_loader}, device, log_wandb)
        evaluation.update({"Model Train Loss":train_loss, 
                           "Model Train Accuracy":train_acc})
        
        # log metrics to wandb
        if log_wandb:
            wandb.log(evaluation)

    torch.save(model.state_dict(), os.path.join("models", "wide_resnet.pth"))