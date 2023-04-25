from torch import Tensor, float32
from torch.nn import Identity
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def PILToScaledTensor():
    # PILToTensor does not scale  but Convert does
    return transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(float32)])


class AutoAugmentedDataset(Dataset):
    def __init__(self, data, labels, classes, nb_steps, batch_size):
        self.data = data
        self.labels = labels
        self.classes = classes
        self.nb_steps = nb_steps
        self.batch_size = batch_size
        # Base augmentation for all data
        #self.transform = transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
        self.transform = Identity() # To do no base augmentation
        self.to_tensor = PILToScaledTensor()

    def __getitem__(self, index):
        # Get the data and label at the specified index
        modulo_index = index%len(self.data)
        img, label = Image.fromarray(self.data[modulo_index]), self.labels[modulo_index]
        img = self.transform(img)
        img = img if isinstance(img, Tensor) else self.to_tensor(img)
        return img, label

    def __len__(self):
        return self.nb_steps*self.batch_size
    
class WeakStrongAugmentDataset(AutoAugmentedDataset):
    def __init__(self, data, classes, weak_transform, strong_transform, batch_size):
        super().__init__(data, labels=None, classes=classes, nb_steps=len(data), batch_size=batch_size)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.classes = classes

    def __getitem__(self, index):
        # Get the data and label at the specified index
        modulo_index = index%len(self.data)
        img = Image.fromarray(self.data[modulo_index])
        img = self.transform(img)
        weakly_transformed_img = self.weak_transform(img)
        strongly_transformed_img = self.strong_transform(img)
        # Checks if tensors already, because it could change with the transformations 
        weakly_transformed_img = weakly_transformed_img if isinstance(weakly_transformed_img, Tensor) else self.to_tensor(weakly_transformed_img)
        strongly_transformed_img = strongly_transformed_img if isinstance(strongly_transformed_img, Tensor) else self.to_tensor(strongly_transformed_img)
        return weakly_transformed_img, strongly_transformed_img

    def __len__(self):
        return self.nb_steps*self.batch_size