from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class AutoAugmentedDataset(Dataset):
    def __init__(self, data, labels, classes):
        self.data = data
        self.labels = labels
        self.classes = classes
        # Base augmentation for all data
        self.transform = transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # Get the data and label at the specified index
        img, label = Image.fromarray(self.data[index]), self.labels[index]
        img = self.transform(img)
        return self.to_tensor(img), label

    def __len__(self):
        return len(self.data)
    
class WeakStrongAugmentDataset(AutoAugmentedDataset):
    def __init__(self, data, classes, weak_transform, strong_transform):
        super().__init__(data, labels=None, classes=classes)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform  
        self.classes = classes

    def __getitem__(self, index):
        # Get the data and label at the specified index
        img = Image.fromarray(self.data[index])
        img = self.transform(img)
        weakly_transformed_img = self.weak_transform(img)
        strongly_transformed_img = self.strong_transform(img)
        return self.to_tensor(weakly_transformed_img), self.to_tensor(strongly_transformed_img)

    def __len__(self):
        return len(self.data)