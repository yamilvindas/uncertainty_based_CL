#!/usr/bin/env python3
"""
    Class defined the memory buffer for replay-based learning.
"""
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

class MemoryBuffer:
    """
        Replay mempry buffer.
    """
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer_x = []
        self.buffer_y = []
        self.seen_samples = 0

    def get_dataset(self):
        if len(self.buffer_x) == 0: return None
        return TensorDataset(torch.stack(self.buffer_x), torch.stack(self.buffer_y))

    def _add_to_buffer(self, x, y):
        self.buffer_x.append(x.cpu())
        self.buffer_y.append(y.cpu())

    def update_uniform(self, new_x, new_y):
        """
            Uniform/Reservoir sampling
        """
        for i in range(len(new_x)):
            if len(self.buffer_x) < self.capacity:
                self._add_to_buffer(new_x[i], new_y[i])
            else:
                j = np.random.randint(0, self.seen_samples)
                if j < self.capacity:
                    self.buffer_x[j] = new_x[i].cpu()
                    self.buffer_y[j] = new_y[i].cpu()
            self.seen_samples += 1

    def update_loss_based(self, new_x, new_y, model, criterion):
        """
            Retain samples with the highest loss (hardest examples)
        """
        model.eval()
        with torch.no_grad():
            outputs = model(new_x.to(self.device))
            losses = criterion(outputs, new_y.to(self.device)).cpu()
        
        # Combine current buffer and new data
        all_x = self.buffer_x + list(new_x.cpu())
        all_y = self.buffer_y + list(new_y.cpu())
        
        # Calculate losses for existing buffer (if not empty)
        if len(self.buffer_x) > 0:
            buffer_outputs = model(torch.stack(self.buffer_x).to(self.device))
            buffer_losses = criterion(buffer_outputs, torch.stack(self.buffer_y).to(self.device)).cpu()
            all_losses = torch.cat((buffer_losses, losses))
        else:
            all_losses = losses

        # Get indices of top 'capacity' losses
        top_indices = torch.topk(all_losses, min(self.capacity, len(all_losses))).indices
        
        self.buffer_x = [all_x[i] for i in top_indices]
        self.buffer_y = [all_y[i] for i in top_indices]

    def update_uncertainty_based(self, new_x, new_y, model, mc_passes=10, mode='epistemic'):
        """
            Monte Carlo Dropout for uncertainty estimation.
            mode: 'epistemic' (model uncertainty) or 'aleatoric' (data uncertainty).
        """
        model.train() # Enable Dropout
        all_x = self.buffer_x + list(new_x.cpu())
        all_y = self.buffer_y + list(new_y.cpu())
        
        uncertainties = []
        inputs = torch.stack(all_x).to(self.device)
        
        with torch.no_grad():
            for i in range(0, len(inputs), 32): 
                batch = inputs[i:i+32]
                
                # Get probabilities for multiple forward passes: Shape (mc_passes, batch_size, num_classes)
                probs = torch.stack([model(batch).softmax(dim=1) for _ in range(mc_passes)])
                
                # 1. Aleatoric Uncertainty: Mean of the entropies
                # Entropy = -sum(p * log(p))
                entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=2) # (mc_passes, batch_size)
                aleatoric = entropies.mean(dim=0) # (batch_size)
                
                # 2. Total Uncertainty: Entropy of the mean probabilities
                mean_probs = probs.mean(dim=0) # (batch_size, num_classes)
                total_uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1) # (batch_size)
                
                # 3. Epistemic Uncertainty: Total - Aleatoric (Mutual Information)
                epistemic = total_uncertainty - aleatoric
                
                if mode == 'epistemic':
                    uncertainties.extend(epistemic.cpu().tolist())
                elif mode == 'aleatoric':
                    uncertainties.extend(aleatoric.cpu().tolist())
                else:
                    raise ValueError("Mode must be 'epistemic' or 'aleatoric'")
                
        uncertainties = torch.tensor(uncertainties)
        # Select samples with the HIGHEST uncertainty
        top_indices = torch.topk(uncertainties, min(self.capacity, len(uncertainties))).indices
        
        self.buffer_x = [all_x[i] for i in top_indices]
        self.buffer_y = [all_y[i] for i in top_indices]

    def update_feature_dissimilarity(self, new_x, new_y, model):
        """
            Greedy K-Center on extracted features to maximize memory diversity
        """
        model.eval()
        all_x = self.buffer_x + list(new_x.cpu())
        all_y = self.buffer_y + list(new_y.cpu())
        
        if len(all_x) <= self.capacity:
            self.buffer_x, self.buffer_y = all_x, all_y
            return

        with torch.no_grad():
            features = model.extract_features(torch.stack(all_x).to(self.device)).cpu()

        # Greedy K-Center
        selected_indices = [np.random.randint(0, len(features))]
        min_distances = torch.norm(features - features[selected_indices[0]], dim=1)

        while len(selected_indices) < self.capacity:
            farthest = torch.argmax(min_distances).item()
            selected_indices.append(farthest)
            dist_to_new = torch.norm(features - features[farthest], dim=1)
            min_distances = torch.min(min_distances, dist_to_new)

        self.buffer_x = [all_x[i] for i in selected_indices]
        self.buffer_y = [all_y[i] for i in selected_indices]


class TensorDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


class LatentVisualizer:
    """
        Allows to visualize the retained replay memory against all the samples
        from all tasks.
    """
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def extract_features_and_labels(self, dataloader):
        self.model.eval()
        features, labels = [], []
        with torch.no_grad():
            for x, y in dataloader:
                if isinstance(y, tuple) or isinstance(y, list): y = y[0]
                feat = self.model.extract_features(x.to(self.device))
                features.append(feat.cpu().numpy())
                labels.append(y.cpu().numpy())
        return np.concatenate(features), np.concatenate(labels)

    def plot_memory_representation(self, task_a_loader, task_b_loader, memory_buffer):
        print("\n\n==========> Extracting features for visualization <==========")
        
        # Extract whole datasets
        feat_A, _ = self.extract_features_and_labels(task_a_loader)
        feat_B, _ = self.extract_features_and_labels(task_b_loader)
        
        # Extract Memory Buffer
        mem_dataset = memory_buffer.get_dataset()
        if mem_dataset is None:
            print("Memory is empty. Cannot plot.")
            return
            
        mem_loader = DataLoader(mem_dataset, batch_size=32, shuffle=False)
        feat_mem, _ = self.extract_features_and_labels(mem_loader)

        # Concatenate everything for t-SNE fitting
        all_features = np.vstack((feat_A, feat_B, feat_mem))
        
        # Generate Labels for coloring: 0 for Task A, 1 for Task B, 2 for Memory
        labels = np.array([0]*len(feat_A) + [1]*len(feat_B) + [2]*len(feat_mem))

        print("\n\n==========> Computing t-SNE (This may take a minute) <==========\n")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latent_2d = tsne.fit_transform(all_features)

        # Split back for plotting
        latent_A = latent_2d[labels == 0]
        latent_B = latent_2d[labels == 1]
        latent_mem = latent_2d[labels == 2]

        plt.figure(figsize=(10, 8))
        
        # Plot full datasets with low alpha (transparency)
        plt.scatter(latent_A[:, 0], latent_A[:, 1], c='blue', alpha=0.3, label='Task A Data', s=10)
        plt.scatter(latent_B[:, 0], latent_B[:, 1], c='green', alpha=0.3, label='Task B Data', s=10)
        
        # Plot memory on top with distinct marker and high visibility
        plt.scatter(latent_mem[:, 0], latent_mem[:, 1], c='red', marker='*', edgecolor='black', 
                    s=150, alpha=1.0, label='Memory Buffer')

        plt.title("t-SNE Projection: Dataset Distribution vs. Replay Memory")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend()
        plt.grid(True)
        plt.show()

