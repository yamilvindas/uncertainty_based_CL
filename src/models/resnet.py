#!/usr/bin/env python3
"""
    ResNet-18 based model
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

class ResNet18CLModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        # Pretrained ResNet
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # Classfier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(base_model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def extract_features(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)