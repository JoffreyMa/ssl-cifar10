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
import wandb
from torch.utils.data.dataset import Subset

def evaluate(models, data_loaders, device, log_wandb):
    evaluation = dict()
    for model_name in models:
        for data_loader_name in data_loaders:
            loss, accuracy, precision, recall, f1, cm = _evaluate(models[model_name], data_loaders[data_loader_name], device, log_wandb)
            prefix = f"{model_name} {data_loader_name}"
            evaluation.update({f"{prefix} Loss":loss, 
                               f"{prefix} Accuracy":accuracy, 
                               f"{prefix} Precision":precision, 
                               f"{prefix} Recall":recall, 
                               f"{prefix} F1":f1, 
                               f"{prefix} Confusion Matrix":cm})
    return evaluation


def _evaluate(model, data_loader, device, log_wandb):
    model.eval()
    loss = 0

    # Store true and predicted labels
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in data_loader:
            y_true.extend(target.cpu().tolist())
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            y_pred.extend(pred.cpu().tolist())

    loss /= len(data_loader.dataset)
    classes = data_loader.dataset.dataset.classes if isinstance(data_loader.dataset, Subset) else data_loader.dataset.classes
    accuracy, precision, recall, f1, cm = metrics(y_true, y_pred, classes, log_wandb)
    return loss, accuracy, precision, recall, f1, cm

def metrics(y_true, y_pred, labels, log_wandb):
    accuracy = accuracy_score(y_true, y_pred)
    # when there are labels for which the classifier didn't predict any samples
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    if log_wandb:
        cm = wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred, class_names=labels)
    else:
        label_mapping = {i:label for i, label in enumerate(labels)}
        y_true_label = [label_mapping[label] for label in y_true]
        y_pred_label = [label_mapping[label] for label in y_pred]
        cm = confusion_matrix(y_true_label, y_pred_label, labels=labels)
    return accuracy, precision, recall, f1, cm
