#!/usr/bin/env python3
"""
    TimeFreq2DCNN — Custom CNN for HITS TCD Doppler Spectrograms

    Architecture designed for 2D time-frequency spectrograms of
    Transcranial Doppler (TCD) ultrasound signals.

    Input : (batch, 3, 96, 224) — RGB spectrogram
    Output: (batch, num_classes) — logits for A / ES / EG

    Structure:
        4 encoder blocks: Conv→BN→ReLU→Conv→BN→ReLU→MaxPool
        Global average pooling: AvgPool2d((12, 28)) [deterministic]
        Classifier head: Dropout → Linear(512→128) → ReLU → Dropout → Linear(128→num_classes)

    Follows the same interface as ResNet18CLModel and SimpleCLModel:
        - forward(x)          → logits
        - extract_features(x) → flat feature vector (used by memory.py
                                 for Feature Dissimilarity and t-SNE)
"""

import torch
import torch.nn as nn


class TimeFreq2DCNNModel(nn.Module):
    """
    CNN for TCD Doppler spectrogram classification (HITS dataset).

    Args:
        num_classes      : number of output classes (3 for HITS: A, ES, EG)
        nb_init_filters  : number of filters in the first conv block
                           (doubled each block: 64 → 128 → 256 → 512)
        dropout_rate     : dropout probability applied in the classifier head
    """

    def __init__(self, num_classes: int = 3,
                 nb_init_filters: int = 64,
                 dropout_rate: float = 0.2):
        super().__init__()

        f = nb_init_filters

        # ── Encoder — 4 conv blocks ───────────────────────────────────────
        self.features = nn.Sequential(

            # Block 1 — (B, 3, 96, 224) → (B, 64, 48, 112)
            nn.Conv2d(3, f, kernel_size=3, padding=1),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f, kernel_size=3, padding=1),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2 — (B, 64, 48, 112) → (B, 128, 24, 56)
            nn.Conv2d(f, f * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f * 2, f * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3 — (B, 128, 24, 56) → (B, 256, 12, 28)
            nn.Conv2d(f * 2, f * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(f * 4, f * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4 + Global pooling — (B, 256, 12, 28) → (B, 512, 1, 1)
            nn.Conv2d(f * 4, f * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(f * 8, f * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(inplace=True),
            # Fixed-size pooling (deterministic backward pass).
            # Input is always (12, 28) after 3x MaxPool2d(2) from (96, 224).
            # Mathematically identical to AdaptiveAvgPool2d((1, 1)).
            nn.AvgPool2d((12, 28)),
        )

        # ── Classifier head ───────────────────────────────────────────────
        # Feature vector: 512-dim (f * 8 = 64 * 8)
        # Mirrors the two-layer head used in ResNet18CLModel and SimpleCLModel.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(f * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def extract_features(self, x):
        """Used directly by Feature Dissimilarity Maximization and t-SNE Plotting"""
        x = self.features(x)
        return torch.flatten(x, 1)