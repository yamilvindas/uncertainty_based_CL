#!/usr/bin/env python3
"""
    Data Handler for HITS (Transcranial Doppler Ultrasound) dataset.

    Dataset: HITS Emboli — HITS-S51-13k-V9k
    Classes: A (Artefact), ES (Solid Emboli), EG (Gas Emboli)
    Format : HDF5 files with PNG paths stored as attributes

    Task A: data_2tasks-src-CLsplit_taskA.hdf5
            Train: 2817 samples | Test: 947 samples
    Task B: data_2tasks-src-CLsplit_taskB.hdf5
            Train: 868 samples  | Test: 139 samples

    Split strategy:
        - Train/Test splits are pre-defined in the HDF5 files.
        - Validation set is carved out from the training split
          using a stratified 85/15 split (same as supervisor protocol).
"""

import os
import random
import sys
from collections import Counter

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset

# For internal imports
sys.path.append(os.path.abspath(os.path.join("..")))
from src.data_processing.OrganMNIST import DataHandler

# =============================================================================
# Internal PyTorch Dataset — reads PNG paths from HDF5
# =============================================================================

class HITSDataset(Dataset):
    """
    PyTorch Dataset for HITS TCD Doppler spectrograms.

    Reads sample metadata (class label + PNG path) from an HDF5 file
    and loads images lazily from disk on __getitem__.

    Args:
        hdf5_path : path to the HDF5 file (Task A or Task B)
        split     : "train" or "test" (HDF5 group key)
        transform : torchvision transform pipeline
    """

    #----- Class attributes and methods -----
    CLASSES     = ["A", "ES", "EG"]
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    #----- Instance attributes and methods -----
    def __init__(self, hdf5_path: str, split: str = "train",
                 transform=None):
        self.hdf5_path = hdf5_path
        self.split     = split
        self.transform = transform
        self.samples   = []   # list of (png_path, label_int)

        self._load_index()

    def _load_index(self):
        """Read all (png_path, label) pairs from HDF5 into memory."""
        group_path = f"mydataset/{self.split}"
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 not found: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, "r") as f:
            if group_path not in f:
                raise KeyError(f"Group '{group_path}' not found in HDF5.")
            group   = f[group_path]
            skipped = 0

            for key in group.keys():
                attrs     = dict(group[key].attrs)
                raw_class = attrs.get("class",   b"")
                png_path  = attrs.get("pngPath", b"")

                if isinstance(raw_class, bytes):
                    raw_class = raw_class.decode("utf-8")
                if isinstance(png_path, bytes):
                    png_path = png_path.decode("utf-8")

                raw_class = raw_class.strip()
                png_path  = png_path.strip()

                if raw_class not in HITSDataset.CLASS_TO_IDX or not png_path:
                    skipped += 1
                    continue

                self.samples.append((png_path, HITSDataset.CLASS_TO_IDX[raw_class]))

            if skipped:
                print(f"  [INFO] Skipped {skipped} invalid samples "
                      f"({self.split})")

        # ── Class distribution summary ───────────────────────────────────
        dist  = Counter(lbl for _, lbl in self.samples)
        total = len(self.samples)
        print(f"  [{self.split}] {total} samples loaded")
        for idx, cname in enumerate(HITSDataset.CLASSES):
            n = dist.get(idx, 0)
            print(f"    {cname}: {n} ({n/total*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        png_path, label = self.samples[idx]
        try:
            img = Image.open(png_path).convert("RGB")
        except Exception as e:
            print(f"  [WARNING] Cannot load {png_path}: {e}")
            img = Image.new("RGB", (224, 96), color=0)
        if self.transform:
            img = self.transform(img)
        return img, label


# =============================================================================
# Stratified Val Split helper
# =============================================================================

def _stratified_split(dataset: HITSDataset, val_pct: float = 0.15,
                      seed: int = 42):
    """
    Carve out val_pct of each class from a HITSDataset to form a
    stratified validation set. Returns (train_subset, val_subset).

    Args:
        dataset : HITSDataset (train split)
        val_pct : fraction to use for validation (default 0.15)
        seed    : random seed for reproducibility
    """
    rng = random.Random(seed)

    class_indices = {}
    for i, (_, lbl) in enumerate(dataset.samples):
        class_indices.setdefault(lbl, []).append(i)

    train_idx, val_idx = [], []
    for lbl, idxs in class_indices.items():
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_pct))
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# =============================================================================
# HITSHandler — follows DataHandler interface
# =============================================================================

class HITSHandler(DataHandler):
    """
    Data handler for the HITS TCD Doppler emboli dataset.

    Follows the same interface as OrganMNISTHandler and CamelyonHandler:
      handler = HITSHandler(batch_size=32, hdf5_a=TASK_A_HDF5.hdf5, hdf5_b=TASK_B_HDF5.hdf5)
      task_a, task_b = handler.get_tasks()
      task_a_train, task_a_val, task_a_test = task_a

    Args:
        batch_size : DataLoader batch size
        val_pct    : fraction of training data reserved for validation
                     (stratified per class, default 0.15 = 15%)
        seed       : random seed for the stratified split
        hdf5_a     : path to Task A HDF5 file (default: TASK_A_HDF5)
        hdf5_b     : path to Task B HDF5 file (default: TASK_B_HDF5)
    """

    def __init__(self,
                hdf5_a: str,
                hdf5_b: str,
                batch_size: int = 32, val_pct: float = 0.15,
                seed: int = 42,
                ):
        super().__init__(batch_size)

        self.val_pct = val_pct
        self.seed    = seed
        self.hdf5_a  = hdf5_a
        self.hdf5_b  = hdf5_b

        # Override base class transform with HITS-specific preprocessing.
        # Input spectrograms are (3, 96, 224) — resize to match model input.
        # ImageNet normalization is used as in OrganMNIST (lite=False) and
        # Camelyon17 handlers.
        self.transform = transforms.Compose([
            transforms.Resize((96, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    # ------------------------------------------------------------------
    # get_tasks — main interface (mirrors OrganMNIST / Camelyon17)
    # ------------------------------------------------------------------

    def get_tasks(self):
        """
        Load Task A and Task B datasets with train/val/test splits.

        Returns:
            task_a : (train_subset, val_subset, test_dataset)
            task_b : (train_subset, val_subset, test_dataset)

        Example:
            handler = HITSHandler(batch_size=32, hdf5_a=TASK_A_HDF5.hdf5, hdf5_b=TASK_B_HDF5.hdf5)
            (a_train, a_val, a_test), (b_train, b_val, b_test) = handler.get_tasks()
        """
        print("\n\n===> Loading HITS TCD Doppler dataset ===")
        print(f"  Task A: {self.hdf5_a}")
        print(f"  Task B: {self.hdf5_b}\n")

        # ── Task A ────────────────────────────────────────────────────────
        print("--- Task A ---")
        task_a_full_train = HITSDataset(self.hdf5_a, "train", self.transform)
        task_a_test       = HITSDataset(self.hdf5_a, "test",  self.transform)
        task_a_train, task_a_val = _stratified_split(
            task_a_full_train, self.val_pct, self.seed)

        # ── Task B ────────────────────────────────────────────────────────
        print("\n--- Task B ---")
        task_b_full_train = HITSDataset(self.hdf5_b, "train", self.transform)
        task_b_test       = HITSDataset(self.hdf5_b, "test",  self.transform)
        task_b_train, task_b_val = _stratified_split(
            task_b_full_train, self.val_pct, self.seed)

        # ── Summary ───────────────────────────────────────────────────────
        self.n_all_train_samples = (len(task_a_full_train) +
                                    len(task_b_full_train))
        print(f"\n\n===> Total number of TRAIN samples for ALL tasks: "
              f"{self.n_all_train_samples}\n")
        print(f"  Task A | train={len(task_a_train)}  "
              f"val={len(task_a_val)}  test={len(task_a_test)}")
        print(f"  Task B | train={len(task_b_train)}  "
              f"val={len(task_b_val)}  test={len(task_b_test)}")

        return (task_a_train, task_a_val, task_a_test), \
               (task_b_train, task_b_val, task_b_test)