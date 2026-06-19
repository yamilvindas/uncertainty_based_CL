#!/usr/bin/env python3
"""
    Data Handler for Camelyon17 dataset.
"""
import os
import sys

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

# For internal imports
sys.path.append(os.path.abspath(os.path.join("..")))
from src.data_processing.OrganMNIST import DataHandler


class CamelyonHandler(DataHandler):
    def __init__(self, batch_size, lite=True):
        super().__init__(batch_size)
        if (lite):
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_tasks(self):
        # Get full dataset
        dataset = get_dataset(dataset="camelyon17", download=True)
        
        # Get train, val, test data
        train_data = dataset.get_subset('train', transform=self.transform)
        val_data = dataset.get_subset('val', transform=self.transform) 
        test_data = dataset.get_subset('test', transform=self.transform)
        
        # Combination of all the data
        all_data = ConcatDataset([train_data, val_data, test_data])
        self.n_all_train_samples = len(train_data)
        print(f"\n\n===> Total number of TRAIN samples for ALL tasks: {self.n_all_train_samples}\n")

        # Get the hospital centers associated to each sample
        centers = []
        for i in range(len(all_data)):
            _, _, metadata = all_data[i]
            centers.append(metadata[0].item()) 
        centers = np.array(centers)

        # Task A: hospitals 0 and 1
        task_a_idx = np.where((centers == 0) | (centers == 1))[0]
        # Task B: hospitals 2 and 3
        task_b_idx = np.where((centers == 2) | (centers == 3))[0]
        # External test: hospitals 4
        ext_test_idx = np.where(centers == 4)[0]

        # Create the datasets for each task
        task_a = Subset(all_data, task_a_idx)
        task_b = Subset(all_data, task_b_idx)
        ext_test = Subset(all_data, ext_test_idx)
        
        # Three-way split: 70% Train, 10% Val, 20% Test
        def split_train_val_test(subset):
            total = len(subset)
            train_size = int(0.7 * total)
            val_size = int(0.1 * total)
            test_size = total - train_size - val_size
            return torch.utils.data.random_split(subset, [train_size, val_size, test_size])

        # Task A: hospitals 0 and 1
        task_a_train, task_a_val, task_a_test = split_train_val_test(task_a)
        # Task B: hospitals 2 and 3
        task_b_train, task_b_val, task_b_test = split_train_val_test(task_b)

        return (task_a_train, task_a_val, task_a_test), (task_b_train, task_b_val, task_b_test), ext_test
    
