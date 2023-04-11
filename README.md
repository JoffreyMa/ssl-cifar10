# ssl-cifar10
## FixMatch with Wide ResNet-28-2 on CIFAR-10

This repository contains the implementation of the FixMatch algorithm combined with the Wide ResNet-28-2 model for semi-supervised learning on the CIFAR-10 dataset.

## Requirements

- Python 3.6 or higher
- PyTorch 1.7.0 or higher
- torchvision 0.8.1 or higher

## Usage

1. Clone the repository:

```bash
git clone https://github.com/JoffreyMa/ssl-cifar10.git
cd ssl-cifar10
```

1. Run the training script:

```bash
python ssl-cifar10/main.py
```

The training script will train the Wide ResNet-28-2 model using the FixMatch algorithm on the CIFAR-10 dataset with 250 randomly selected labeled images. The test loss and test accuracy will be printed for each epoch, and the trained model will be saved as fixmatch_wide_resnet.pth.

## Files

* wide_resnet.py: Contains the implementation of the Wide ResNet-28-2 model.
* fixmatch.py: Contains the implementation of the FixMatch algorithm.
* utils.py: Contains utility functions for creating data loaders.
* main.py: The main script to run the training and evaluation.

## References

* Sohn, K., Berthelot, D., Li, C., Zhang, Z., Carlini, N., Cubuk, E. D., Kurakin, A., Zhang, H., & Raffel, C. (2020). FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. arXiv preprint arXiv:2001.07685.
* Zagoruyko, S., & Komodakis, N. (2016). Wide Residual Networks. arXiv preprint arXiv:1605.07146.
* Chaudhary. The Illustrated FixMatch for Semi-Supervised Learning. https://amitness.com/2020/03/fixmatch-semi-supervised/