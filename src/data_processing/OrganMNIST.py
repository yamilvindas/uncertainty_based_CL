#!/usr/bin/env python3
"""
    Data handler for OrganMNIST dataset.
"""
# Dataset specific imports
import medmnist
import numpy as np
import torch
import torchvision.transforms as transforms
from medmnist import INFO


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
    #----- Class attributes and methods -----
    IDX_TO_CLASS = None
    
    @classmethod
    def load_class_mapping(cls):
        if cls.IDX_TO_CLASS is None:
            info = INFO['organamnist']
            cls.IDX_TO_CLASS = {int(i): class_name for i, class_name in info['label'].items()}
    
    #----- Instance attributes and methods -----
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
        
        # Load the class mapping if not already loaded
        self.load_class_mapping()

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
        
        # Total number of samples
        self.n_all_train_samples = len(task_a_train) + len(task_b_train)
        print(f"\n\n===> Total number of TRAIN samples for ALL tasks: {self.n_all_train_samples}\n")

        return (task_a_train, task_a_val, task_a_test), (task_b_train, task_b_val, task_b_test)
    
