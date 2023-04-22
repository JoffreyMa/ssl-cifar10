# FixMatch algorithm implementation

import torch
import torch.nn.functional as F
from ema import EMA
from utils import save_transformed_images

class FixMatch:
    def __init__(self, model, device, optimizer, scheduler, labeled_loader, unlabeled_loader, test_loader, lambda_u=1, threshold=0.95, ema_decay=0.999, check_transformations=False, transformed_img_dir="transformed_images"):
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
        self.check_transformations = check_transformations
        self.transformed_img_dir = transformed_img_dir

    def train(self):
        train_loss = 0
        train_loss_x = 0
        train_loss_u = 0
        train_pct_above_thresh = 0
        ratio_unlabeled_labeled = 0

        self.model.train()
        for i, ((inputs_x, targets_x), (inputs_u_weakly_augmented, inputs_u_strongly_augmented)) in enumerate(zip(self.labeled_loader, self.unlabeled_loader)):
            # Save first batch for each type to check augmentations
            if i==0:
                ratio_unlabeled_labeled = len(inputs_u_weakly_augmented)/len(inputs_x)
                if self.check_transformations:
                    save_transformed_images(inputs_x, output_dir=self.transformed_img_dir, prefix=f"input_x")
                    save_transformed_images(inputs_u_weakly_augmented, output_dir=self.transformed_img_dir, prefix=f"input_u_weakly_augmented")
                    save_transformed_images(inputs_u_strongly_augmented, output_dir=self.transformed_img_dir, prefix=f"input_u_strongly_augmented")
                
            # Transfer to device
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
            train_loss_x += loss_x
            train_loss_u += loss_u
            # Accumulate the samples with confidence above threshold
            train_pct_above_thresh += torch.mean(mask).cpu().tolist()

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Update the EMA model
            self.ema.update(self.model)
            
            # Increment step counter for scheduler
            self.current_step += 1

        # Return metrics to follow
        # loss over the combination of labeled and unlabeled data
        # loss for the labeled part
        # loss for the unlabeled part
        # part of samples with confidence above threshold 
        train_loss /= len(self.labeled_loader.dataset) + self.lambda_u * ratio_unlabeled_labeled * len(self.labeled_loader.dataset)
        train_loss_x /= len(self.labeled_loader.dataset)
        train_loss_u /= ratio_unlabeled_labeled * len(self.labeled_loader.dataset)
        train_pct_above_thresh /= ratio_unlabeled_labeled * len(self.labeled_loader.dataset)
        return train_loss, train_loss_x, train_loss_u, train_pct_above_thresh