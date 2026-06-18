#!/usr/bin/env python3
"""
    Data handler for OrganMNIST dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

import numpy as np
from sklearn.metrics import accuracy_score
import optuna
import copy

# Dataset specific imports
import medmnist
from medmnist import INFO
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader

class DataHandler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_class_weights(self, dataset):
        labels = [y for _, y in dataset]
        labels = np.array(labels).flatten()
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)

class OrganMNISTHandler(DataHandler):
    def __init__(self, batch_size, lite=True):
        super().__init__(batch_size)
        if (lite):
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ]) 
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _load_medmnist(self, data_flag, split):
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split=split, transform=self.transform, download=True)
        dataset.labels = dataset.labels.squeeze() 
        return dataset

    def get_tasks(self):
        # Native MedMNIST Train/Val/Test splits
        # Task A: OrganAMNIST
        task_a_train = self._load_medmnist('organamnist', 'train')
        task_a_val = self._load_medmnist('organamnist', 'val')
        task_a_test = self._load_medmnist('organamnist', 'test')
        
        # Task B: OrganCMNIST
        task_b_train = self._load_medmnist('organcmnist', 'train')
        task_b_val = self._load_medmnist('organcmnist', 'val')
        task_b_test = self._load_medmnist('organcmnist', 'test')
        
        return (task_a_train, task_a_val, task_a_test), (task_b_train, task_b_val, task_b_test)
    
