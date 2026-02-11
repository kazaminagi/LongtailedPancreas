"""
Long-tailed CIFAR Dataloader for IBC-EDL
Returns (image, label, index) for training to support IBC's soft label generation
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import json
import os


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


class LongTailedCIFAR(Dataset):
    """Long-tailed CIFAR dataset with index return for IBC"""
    
    def __init__(self, dataset, imb_type, imb_factor, root_dir, transform, mode):
        self.transform = transform
        self.mode = mode
        self.dataset = dataset
        
        if dataset == 'cifar10':
            base_folder = 'cifar-10-batches-py'
            self.num_classes = 10
        else:
            base_folder = 'cifar-100-python'
            self.num_classes = 100
            
        file_path = os.path.join(root_dir, base_folder)
        
        if mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle(f'{file_path}/test_batch')
                self.data = test_dic['data']
                self.targets = test_dic['labels']
            else:
                test_dic = unpickle(f'{file_path}/test')
                self.data = test_dic['data']
                self.targets = test_dic['fine_labels']
            self.data = self.data.reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
        else:
            # Load training data
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = f'{file_path}/data_batch_{n}'
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label += data_dic['labels']
                train_data = np.concatenate(train_data)
            else:
                train_dic = unpickle(f'{file_path}/train')
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            
            train_data = train_data.reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
            self.data = train_data
            self.targets = np.array(train_label)
            
            # Generate long-tailed distribution
            self.img_num_list = self._get_img_num_per_cls(imb_type, imb_factor)
            
            # Create or load imbalanced indices
            os.makedirs(os.path.join(file_path, 'longtail_file'), exist_ok=True)
            imb_file = os.path.join(file_path, 'longtail_file', 
                                   f'cifar{self.num_classes}_{imb_type}_{imb_factor}')
            self._gen_imbalanced_data(imb_file)
    
    def _get_img_num_per_cls(self, imb_type, imb_factor):
        img_max = len(self.data) / self.num_classes
        img_num_per_cls = []
        
        if imb_type == 'exp':
            for cls_idx in range(self.num_classes):
                num = img_max * (imb_factor ** (cls_idx / (self.num_classes - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(self.num_classes // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(self.num_classes // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls = [int(img_max)] * self.num_classes
            
        return img_num_per_cls
    
    def _gen_imbalanced_data(self, imb_file):
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        
        self.num_per_cls_dict = {}
        for cls, num in zip(classes, self.img_num_list):
            self.num_per_cls_dict[cls] = num
        
        if os.path.exists(imb_file):
            imb_sample = json.load(open(imb_file, "r"))
        else:
            imb_sample = []
            for cls, num in zip(classes, self.img_num_list):
                idx = np.where(targets_np == cls)[0]
                np.random.shuffle(idx)
                imb_sample.extend(idx[:num].tolist())
            json.dump(imb_sample, open(imb_file, 'w'))
            print(f"Saved imbalanced indices to {imb_file}")
        
        imb_sample = np.array(imb_sample)
        self.data = self.data[imb_sample]
        self.targets = self.targets[imb_sample]
    
    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        img = self.transform(img)
        target = int(self.targets[index])
        
        if self.mode == 'test':
            return img, target
        else:
            return img, target, index
    
    def __len__(self):
        return len(self.data)
    
    def get_cls_num_list(self):
        return [self.num_per_cls_dict[i] for i in range(self.num_classes)]


def get_dataloader(dataset='cifar10', imb_type='exp', imb_factor=0.01,
                   batch_size=64, num_workers=4, root_dir='/mnt/zzh'):
    """
    Create train and test dataloaders for long-tailed CIFAR
    
    Returns:
        train_loader: DataLoader with (img, label, index)
        test_loader: DataLoader with (img, label)
        cls_num_list: list of sample counts per class
    """
    # CIFAR normalization
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.507, 0.487, 0.441)
        std = (0.267, 0.256, 0.276)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_dataset = LongTailedCIFAR(
        dataset=dataset,
        imb_type=imb_type,
        imb_factor=imb_factor,
        root_dir=root_dir,
        transform=train_transform,
        mode='train'
    )
    
    test_dataset = LongTailedCIFAR(
        dataset=dataset,
        imb_type=imb_type,
        imb_factor=imb_factor,
        root_dir=root_dir,
        transform=test_transform,
        mode='test'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    cls_num_list = train_dataset.get_cls_num_list()
    
    print(f"Dataset: {dataset}, Imbalance: {imb_type}_{imb_factor}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Class distribution: {cls_num_list}")
    
    return train_loader, test_loader, cls_num_list
