# Evaluate performances of a model on a multiclass classification task 

import torch
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import torch.nn.functional as F

def evaluate(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Store true and predicted labels
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in test_loader:
            y_true.extend(target.cpu().numpy())
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            y_pred.extend(pred.cpu().numpy())

    test_loss /= total
    accuracy = 100.0 * correct / total
    accuracy, precision, recall, f1 = metrics(y_true, y_pred)
    #cm = confusion_matrix(y_true, y_pred, test_loader.dataset.classes)

    return test_loss, accuracy, precision, recall, f1#, cm

def metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, precision, recall, f1

def confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm
