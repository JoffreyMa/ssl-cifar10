# Main script to run the training and evaluation process.

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from wide_resnet import WideResNet
from fixmatch import FixMatch
from utils import create_data_loaders, download_cifar10
from evaluate import evaluate
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR


def main():
    # Parameters
    # Optimizer
    lr = 0.03
    momentum = 0.9
    weight_decay = 5e-4
    # Training
    nb_epochs = 1024
    nb_steps = 1024
    # Dataloader
    batch_size=64
<<<<<<< HEAD
    ratio_unlabeled_labeled=7
    seed=5
=======
    ratio_unlabeled_labeled=3
    seed=42
>>>>>>> 0a80fda16b06462aa83e3280401c6c9082cfb35d
    # Model
    depth=28 
    widen_factor=2
    dropout_rate=0.3
    # Goal
    num_classes=10
    # FixMatch
    lambda_u=1
    threshold=0.35
    # EMA
    ema_decay=0.333
    # Log
    log_wandb = True
    check_transformations = False

    # Set device for computation
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Download data if necessary
    download_cifar10()
    
    # Instanciate train and test sets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
    
    # Instanciate data loaders for WideResNet with FixMatch
    labeled_loader, unlabeled_loader, test_loader = create_data_loaders(trainset, testset, batch_size=batch_size, ratio_unlabeled_labeled=ratio_unlabeled_labeled, nb_steps=nb_steps, seed=seed)
    
    # Declare WideResNet
    model = WideResNet(depth=depth, widen_factor=widen_factor, dropout_rate=dropout_rate, num_classes=num_classes)
    model.to(device)
    
    # Declare the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    # Create the learning rate scheduler
    # Differ from the fixmatch paper scheduler but is in the same spirit
    total_training_steps = len(labeled_loader) * nb_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps, eta_min=0, last_epoch=-1)

    
    # Declare the FixMatch
    fixmatch = FixMatch(model, device, optimizer, scheduler, labeled_loader, unlabeled_loader, test_loader, lambda_u, threshold, ema_decay, check_transformations)

    # start a new wandb run to track this script
    if log_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="ssl-cifar10",
            
            # track hyperparameters and run metadata
            config={
            "architecture": "WideResNet-28-2",
            "dataset": "CIFAR-10",
            "epochs": nb_epochs,
            "learning_rate": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "ratio_unlabeled_labeled": ratio_unlabeled_labeled,
            "seed": seed,
            "depth": depth,
            "widen_factor": widen_factor,
            "dropout_rate": dropout_rate,
            "lambda_u": lambda_u,
            "threshold": threshold,
            "ema_decay": ema_decay
            }
        )
    
    for _ in range(nb_epochs):
        train_loss, train_loss_x, train_loss_u, train_pct_above_thresh = fixmatch.train()

        # Evaluate with both ExponentialMovingAverage model and original model
        models = {"Model":model, "EMA_Model":fixmatch.ema.ema_model}
        # Evaluate on data with available labels
        data_loaders = {"Test":test_loader, "Labeled":labeled_loader}
        evaluation = evaluate(models, data_loaders, device, log_wandb)
        evaluation.update({"Model Train Loss":train_loss, 
                           "Model Train_Labeled Loss":train_loss_x, 
                           "Model Train_Unlabeled Loss":train_loss_u, 
                           "Model Train_Unlabeled Pct_above_threshold":train_pct_above_thresh,
                           "Learning rate":optimizer.param_groups[0]["lr"],
                           "Current step": fixmatch.current_step})
        
        # log metrics to wandb
        if log_wandb:
            wandb.log(evaluation)
        else:
            print(f"Train Loss: {train_loss}, \
                  Train PctAboveThreshold: {train_pct_above_thresh}, \
                  Test Loss: {evaluation['Model Test Loss']}, \
                  Test Accuracy: {evaluation['Model Test Accuracy']}")
            
    torch.save(model.state_dict(), os.path.join("models", "fixmatch_wide_resnet_perso.pth"))

if __name__ == "__main__":
    main()
