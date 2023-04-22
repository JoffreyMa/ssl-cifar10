# FixMatch algorithm implementation

import torch
import torch.nn.functional as F
from ema import EMA

class FixMatch:
    def __init__(self, model, device, optimizer, scheduler, labeled_loader, unlabeled_loader, test_loader, lambda_u=1, threshold=0.95, ema_decay=0.999):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.test_loader = test_loader
        self.lambda_u = lambda_u
        self.threshold = threshold
        self.current_step = 0
        self.ema = EMA(model, decay=ema_decay, device=device)

    def train(self):
        train_loss = 0
        self.model.train()
        for (inputs_x, targets_x), (inputs_u_weakly_augmented, inputs_u_strongly_augmented) in zip(self.labeled_loader, self.unlabeled_loader):
            inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
            inputs_u_weakly_augmented, inputs_u_strongly_augmented = inputs_u_weakly_augmented.to(self.device).float(), inputs_u_strongly_augmented.to(self.device).float()
            
            # Reset gradient
            self.optimizer.zero_grad()

            # Forward pass for labeled data
            logits_x = self.model(inputs_x)
            loss_x = F.cross_entropy(logits_x, targets_x)

            # Forward pass for unlabeled data
            logits_u_weakly_augmented = self.model(inputs_u_weakly_augmented)
            logits_u_strongly_augmented = self.model(inputs_u_strongly_augmented)

            # Apply the sharpening function
            max_probs, pseudo_labels = torch.max(logits_u_weakly_augmented, dim=-1)
            
            # Select only samples with confidence above the threshold
            mask = max_probs.ge(self.threshold).float()

            # Compute the consistency loss
            loss_u = F.cross_entropy(logits_u_strongly_augmented, pseudo_labels, reduction='none')
            loss_u = (loss_u * mask).mean()

            # Combine the labeled and unlabeled loss
            loss = loss_x + self.lambda_u * loss_u

            # Accumulate train loss
            train_loss += loss

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Update the EMA model
            self.ema.update(self.model)
            
            # Increment step counter for scheduler
            self.current_step += 1

        # Return loss over the combination of labeled and unlabeled data
        train_loss /= len(self.labeled_loader.dataset)
        return train_loss