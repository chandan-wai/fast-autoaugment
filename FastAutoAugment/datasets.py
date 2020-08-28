import torchvision
from collections import defaultdict
from PIL import Image
from theconf import Config as C


class CIFAR10_mod(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.hardness_scores = defaultdict(defaultdict)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        measure = C.get()['hardness']['aug_measure']
        if measure in self.hardness_scores and index in self.hardness_scores[measure]:
            hardness_score = self.hardness_scores[measure][index]
        else:
            hardness_score = 1
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            for t in self.transform.transforms:
                from FastAutoAugment.data import Augmentation, CutoutDefault
                if isinstance(t, (Augmentation, CutoutDefault)):
                    img = t(img, hardness_score)
                else:
                    img = t(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    
class CIFAR100_mod(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.hardness_scores = defaultdict(defaultdict)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        measure = C.get()['hardness']['aug_measure']
        if measure in self.hardness_scores and index in self.hardness_scores[measure]:
            hardness_score = self.hardness_scores[measure][index]
        else:
            hardness_score = 1
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            for t in self.transform.transforms:
                from FastAutoAugment.data import Augmentation, CutoutDefault
                if isinstance(t, (Augmentation, CutoutDefault)):
                    img = t(img, hardness_score)
                else:
                    img = t(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    
class SVHN_mod(torchvision.datasets.SVHN):
    def __init__(self, root, split, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, split, transform, target_transform, download)
        self.hardness_scores = defaultdict(defaultdict)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        measure = C.get()['hardness']['aug_measure']
        if measure in self.hardness_scores and index in self.hardness_scores[measure]:
            hardness_score = self.hardness_scores[measure][index]
        else:
            hardness_score = 1
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            for t in self.transform.transforms:
                from FastAutoAugment.data import Augmentation, CutoutDefault
                if isinstance(t, (Augmentation, CutoutDefault)):
                    img = t(img, hardness_score)
                else:
                    img = t(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index