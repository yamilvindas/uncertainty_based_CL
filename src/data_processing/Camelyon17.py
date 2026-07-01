#!/usr/bin/env python3
"""
    Data Handler for Camelyon17 dataset.
"""
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Subset

# Dataset specific imports
from wilds import get_dataset
from wilds.common.grouper import CombinatorialGrouper

# For internal imports
sys.path.append(os.path.abspath(os.path.join("..")))
from src.data_processing.OrganMNIST import DataHandler


class CamelyonHandler(DataHandler):
    """Loads the Camelyon17 dataset and provides train/val/test splits for continual learning tasks.
    y is binary. It is 1 if the central 32x32 region contains any tumor tissue, and 0 otherwise.
    """
    #----- Class attributes and methods -----
    CLASSES = ["Normal", "Tumor"]
    IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
    
    #----- Instance attributes and methods -----
    def __init__(self, batch_size, lite=True, samples_per_stratum=2500, cache_dir="./data/camelyon17_v1.0/"):
        # NOTE: Added samples_per_stratum and cache_dir to manage undersampling size and file saving
        super().__init__(batch_size)
        self.samples_per_stratum = samples_per_stratum
        self.cache_dir = cache_dir
        
        if (lite):
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _undersample_train(self, train_subset, all_data, centers, labels, task_name):
        """
            Helper method to load or generate stratified undersampled indices.
        """
        # Path to the file where we will save/load the undersampled indices
        file_path = os.path.join(self.cache_dir, f"camelyon17_{task_name}_train_indices.pt")
        
        if os.path.exists(file_path):
            print(f"===> Loading cached undersampled indices for {task_name} from {file_path} <===\n")
            final_abs_indices = torch.load(file_path)
        else:
            print(f"===> Generating stratified undersampled indices for {task_name} <===\n")
            
            # Map the nested Subset indices (train_subset -> task_subset) 
            # back to the absolute indices of all_data
            abs_indices = np.array([train_subset.dataset.indices[i] for i in train_subset.indices])
            
            # Getting all the possible centers and labels
            task_centers = centers[abs_indices]
            task_labels = labels[abs_indices]
            final_abs_indices = []
            
            # Stratify by 0-indexed center and tumor label
            strata = list(set(zip(task_centers, task_labels)))
            for c, l in strata:
                mask = (task_centers == c) & (task_labels == l)
                matching_abs_indices = abs_indices[mask]
                
                if (len(matching_abs_indices) > self.samples_per_stratum):
                    sampled = np.random.choice(matching_abs_indices, size=self.samples_per_stratum, replace=False)
                else:
                    sampled = matching_abs_indices
                    
                final_abs_indices.extend(sampled.tolist())
                
            # Save the absolute indices for future CL experiments
            torch.save(final_abs_indices, file_path)
            print(f"\n===> Saved {len(final_abs_indices)} indices to {file_path} <===\n")
            
        # Return a new Subset wrapping all_data with our custom indices
        return Subset(all_data, final_abs_indices)
    
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

        # NOTE: Optimization: replace the slow __getitem__ loop with vectorized metadata extraction
        # NOTE: Also extracting labels for accurate stratification
        print("===> Extracting metadata...")
        all_metadata = torch.cat([
            train_data.metadata_array, 
            val_data.metadata_array, 
            test_data.metadata_array
        ])
        centers = all_metadata[:, 0].numpy()  # Hospital mapping (0-indexed: 0,1,2,3,4)
        labels = all_metadata[:, 1].numpy()   # Tumor vs Normal

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
        
        # NOTE: Added task_name parameter and random seed generator
        def split_train_val_test(subset, task_name):
            total = len(subset)
            train_size = int(0.7 * total)
            val_size = int(0.1 * total)
            test_size = total - train_size - val_size
            
            # Use a fixed generator to lock the 70/10/20 boundaries
            generator = torch.Generator().manual_seed(42)
            train_sub, val_sub, test_sub = torch.utils.data.random_split(subset, [train_size, val_size, test_size], generator=generator)
            
            # Apply undersampling ONLY to the train split, leaving val/test untouched
            train_sub = self._undersample_train(train_sub, all_data, centers, labels, task_name)
            
            return train_sub, val_sub, test_sub

        # Task A: hospitals 0 and 1
        task_a_train, task_a_val, task_a_test = split_train_val_test(task_a, "TaskA")
        # Task B: hospitals 2 and 3
        task_b_train, task_b_val, task_b_test = split_train_val_test(task_b, "TaskB")

        return (task_a_train, task_a_val, task_a_test), (task_b_train, task_b_val, task_b_test), ext_test
