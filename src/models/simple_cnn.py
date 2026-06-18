#!/usr/bin/env python3
"""
    CNN (OrganMNIST ) and MobileNetV3 (Camelyon17)
"""
import torch
import torch.nn as nn
import torchvision.models as models

class SimpleCLModel(nn.Module):
    def __init__(self, model_type, num_classes, dropout_rate=0.5):
        super().__init__()
        self.model_type = model_type
        
        if (model_type.lower() == "organcnn"):
            # Native OrganMNIST: 1 channel input, 28x28 spatial dimension
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2), # Output: 32 x 14 x 14
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2), # Output: 64 x 7 x 7
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)) # Output: 128 x 1 x 1
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes)
            )
            
        elif (model_type.lower() == "camelyonmobilenet"):
            # Initialized WITHOUT weights to be trained completely from scratch
            base_model = models.mobilenet_v3_small(weights=None)
            
            # Extract feature maps up until the pooling layer
            self.features = nn.Sequential(
                base_model.features,
                base_model.avgpool
            )
            # Adapt classifier to match requested classes and dropout rate
            in_features = base_model.classifier[0].in_features
            self.classifier = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.Hardswish(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def extract_features(self, x):
        """Used directly by Feature Dissimilarity Maximization and t-SNE Plotting"""
        x = self.features(x)
        return torch.flatten(x, 1)