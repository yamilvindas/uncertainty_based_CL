#!/usr/bin/env python3
"""
    Class defined the memory buffer for replay-based learning.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

from src.data_processing.Camelyon17 import CamelyonHandler
from src.data_processing.HITS import HITSDataset
from src.data_processing.OrganMNIST import OrganMNISTHandler


class MemoryBuffer:
    """
        Replay memory buffer.
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
        #print(f"\n\n==========> Uniform memory update <==========\n\n")
        for i in range(len(new_x)):
            if len(self.buffer_x) < self.capacity:
                self._add_to_buffer(new_x[i], new_y[i])
            else:
                j = np.random.randint(0, self.seen_samples)
                if j < self.capacity:
                    self.buffer_x[j] = new_x[i].cpu()
                    self.buffer_y[j] = new_y[i].cpu()
            self.seen_samples += 1

    def update_loss_based(
                            self,
                            new_x,
                            new_y,
                            model,
                            criterion,
                            uniform_ratio=0.5
                        ):
        """
            Hybrid sampling: Retain X% uniformly, and the rest based on highest loss (hardest examples)
        """
        #print(f"\n\n==========> Loss-based memory update <==========\n\n")
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

        # Get the total number of sampls to process
        pool_size = len(all_x)
        target_capacity = min(self.capacity, pool_size)
        
        # If we haven't reached capacity, just keep everything
        if (pool_size <= target_capacity):
            self.buffer_x = all_x
            self.buffer_y = all_y
            return

        # Calculate exact counts for the split
        num_uniform = int(target_capacity * uniform_ratio)
        num_strategic = target_capacity - num_uniform
        all_indices = np.arange(pool_size)

        # Uniform Selection
        uniform_indices = np.random.choice(all_indices, num_uniform, replace=False)

        # Strategic (Loss-based) Selection from the REMAINING candidates
        remaining_mask = np.ones(pool_size, dtype=bool)
        remaining_mask[uniform_indices] = False
        remaining_indices = all_indices[remaining_mask]
        remaining_losses = all_losses[remaining_indices]

        # Get indices of top 'num_strategic' losses from the remaining pool
        top_k_relative = torch.topk(remaining_losses, num_strategic).indices
        strategic_indices = remaining_indices[top_k_relative.numpy()]

        # Combine both selections
        final_selected_indices = np.concatenate([uniform_indices, strategic_indices])
        
        self.buffer_x = [all_x[i] for i in final_selected_indices]
        self.buffer_y = [all_y[i] for i in final_selected_indices]

    def update_uncertainty_based(
                                    self,
                                    new_x,
                                    new_y,
                                    model,
                                    we=1.0,
                                    wH=0.1,
                                    wa=0.1,
                                    alea_drop_fraction=0.15,
                                    mc_passes=10,
                                    uniform_ratio=0.5,
                                    by_class = False,
                                    class_distribution = None
                                ):
        """
            Hybrid sampling: Retain X% uniformly, and the rest using advanced uncertainty curation 
            balancing Epistemic exploration, Entropy maximization, and Aleatoric noise mitigation.
            
            Parameters:
            - we (float): weight for epistemic uncertainty
            - wH (float): weight for predictive entropy (or total uncertainty)
            - wa (float): weight for aleatoric uncertainty
            - alea_drop_fraction (float): Fraction of samples to drop based on high aleatoric uncertainty, to avoid outliers
            - mc_passes (int): Number of Monte Carlo forward passes for uncertainty estimation
            - uniform_ratio (float): Ratio of samples to retain uniformly
            - by_class (bool): If True, perform uncertainty-based selection per class to ensure class balance in the memory buffer.
            - class_distribution (dict): A dictionary mapping class labels to their respective proportions in the current task training set. 
            This is used to determine the proportion of samples to retain per class when by_class is True.
        """
        #print(f"\n\n==========> Uncertainty-based memory update <==========\n\n")
        #====================================================================================================#
        #====================================================================================================#
        # Selectively turn on Dropout layers for the Monte Carlo passes
        # NOTE: we do not use model.train() to avoid updating the mean of batch norm layers if they exists !
        def enable_dropout(m):
            if type(m) == torch.nn.Dropout or type(m) == torch.nn.Dropout2d:
                m.train()
        model.apply(enable_dropout)

        # Define full set of samples (samples n memory + current batch of incoming samples)
        all_x = self.buffer_x + list(new_x.cpu())
        all_y = self.buffer_y + list(new_y.cpu())

        #====================================================================================================#
        #====================================================================================================#
        # Uncertainties computation (per sample)
        inputs = torch.stack(all_x).to(self.device)

        raw_epistemic = []
        raw_aleatoric = []
        raw_entropy = []

        # Gather raw uncertainty profiles across all combined samples
        with torch.no_grad():
            for i in range(0, len(inputs), 32): 
                batch = inputs[i:i+32]
                
                # Get probabilities for multiple forward passes: Shape (mc_passes, batch_size, num_classes)
                probs = torch.stack([model(batch).softmax(dim=1) for _ in range(mc_passes)])
                
                # Aleatoric Uncertainty: Mean of the entropies
                # Entropy = -sum(p * log(p))
                entropies_per_pass = -torch.sum(probs * torch.log(probs + 1e-10), dim=2) # (mc_passes, batch_size)
                aleatoric = entropies_per_pass.mean(dim=0) # (batch_size)

                # Total Entropy (Entropy of the average predictive probability)
                mean_probs = probs.mean(dim=0) # (batch_size, num_classes)
                total_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1) # (batch_size)
                
                # Epistemic Uncertainty (Mutual Information = Total - Aleatoric)
                epistemic = total_entropy - aleatoric
                
                # Add to raw uncertainties
                raw_epistemic.extend(epistemic.cpu().numpy())
                raw_aleatoric.extend(aleatoric.cpu().numpy())
                raw_entropy.extend(total_entropy.cpu().numpy())

        # Get the uncertainties
        ua_mean = np.array(raw_aleatoric)
        ue_mean = np.array(raw_epistemic)
        h_mean = np.array(raw_entropy)
        
        #====================================================================================================#
        #====================================================================================================#
        # Min-Max Normalization Helper
        def min_max_normalize(arr):
            denominator = arr.max() - arr.min()
            if (denominator == 0):
                return np.zeros_like(arr)
            return (arr - arr.min()) / denominator
        # MinMax normalization
        epistemic_norm = min_max_normalize(ue_mean)
        aleatoric_norm = min_max_normalize(ua_mean)
        entropy_norm = min_max_normalize(h_mean)
        
        #====================================================================================================#
        #====================================================================================================#
        # Compute Composite Curation Scores
        # NOTE: epistemic measures model ignorance, so this term ensures the memory buffer captures the diversity of the data manifold, pulling in rare edge cases and minority groups.
        # NOTE: aleatoric measures data noise, so this term tries to reduce data noise in the memory.
        # NOTE: the predictive entropy (similar or equivalent to total uncertainty) acts as the "Decision Boundary" anchor, as it measures the flatness of the output probabilities. This term is important as forgetting happens at the boundaries, entropy finds the boundaries to perform hard mining.
        scores = we * epistemic_norm + wH * entropy_norm - wa * aleatoric_norm
        
        #====================================================================================================#
        #====================================================================================================#
        # Determine number of samples to keep
        pool_size = len(all_x)
        target_capacity = min(self.capacity, pool_size)
        
        # If we haven't reached capacity, just keep everything
        if (pool_size <= target_capacity):
            self.buffer_x = all_x
            self.buffer_y = all_y
            return

        # If capacity is reached, sample based on a ratio of uniform and uncertainty-score sampling
        final_selected_indices = []
        if by_class:
            assert class_distribution is not None, "Class_distribution must be provided when by_class is True"
            available_classes = list(class_distribution.keys())
            cum_class_target_capacity = 0
            for i_c, c in enumerate(available_classes):
                class_mask = (np.array(all_y) == c)
                class_indices = np.where(class_mask)[0]
                
                if i_c < len(available_classes) - 1:
                    class_target_capacity = round(class_distribution[c] * target_capacity) # We need to pick class_ratio * target_capacity samples for this class
                else:
                    # Last class takes the remaining capacity
                    class_target_capacity = target_capacity - cum_class_target_capacity

                class_selected_indices = self._get_uncertainty_based_indices(class_indices, scores[class_indices], ua_mean[class_indices], class_target_capacity, uniform_ratio, alea_drop_fraction)
                final_selected_indices.extend(class_selected_indices)
                
                cum_class_target_capacity += len(class_selected_indices) # Keep the real count of selected samples, if class_target_capacity was not attainable
                
        else:
            final_selected_indices = self._get_uncertainty_based_indices(np.arange(pool_size), scores, ua_mean, target_capacity, uniform_ratio, alea_drop_fraction)
        # Commit chosen tensors back to storage
        self.buffer_x = [all_x[i] for i in final_selected_indices]
        self.buffer_y = [all_y[i] for i in final_selected_indices]

    def _get_uncertainty_based_indices(self, candidate_indices, scores, ua_mean, candidate_target_capacity, uniform_ratio, alea_drop_fraction):
        """Based on a list of indices and their corresponding uncertainty scores:
        - drop the indices with the highest aleatoric uncertainty (to avoid noisy outliers)
        - select a fraction of the remaining indices uniformly
        - select the rest based on the highest uncertainty scores

        Args:
            candidate_indices (np.array): A list of indices of candidate samples
            scores (np.array): An array of uncertainty scores for the candidate samples
            ua_mean (np.array): An array of aleatoric uncertainties for the candidate samples
            candidate_target_capacity (int): The target capacity for the selected samples
            uniform_ratio (float): The ratio of uniformly selected samples
            alea_drop_fraction (float): The fraction of noisy samples (in terms of aleatoric uncertainty) to drop

        Returns:
            np.array: A list of indices of the selected samples
        """
        # Calculate exact counts for the split
        num_candidate = len(candidate_indices)
        num_uniform = int(candidate_target_capacity * uniform_ratio)
        num_strategic = candidate_target_capacity - num_uniform
        
        # Second check that number of candidates is sufficient to meet the target capacity, else select all samples
        if (num_candidate <= candidate_target_capacity):
            return candidate_indices
        
        # Uniform Selection
        # uniform_indices = np.random.choice(candidate_indices, num_uniform, replace=False)
        uniform_idx = np.random.choice(np.arange(num_candidate), num_uniform, replace=False)
        uniform_sample_indices = candidate_indices[uniform_idx]

        # Create mask to ONLY consider the remaining unpicked samples for strategy
        remaining_mask = np.ones(num_candidate, dtype=bool)
        remaining_mask[uniform_idx] = False
        
        # Filter out highly noisy (high aleatoric) outliers using percentile logic
        # NOTE the drop is done on all the candidates, whatever if they were uniformly chosen or not.
        max_droppable = num_candidate - candidate_target_capacity
        intended_drop = int(num_candidate * alea_drop_fraction)
        actual_drop = min(intended_drop, max_droppable)
        
        if (actual_drop > 0):
            keep_percentage = 100.0 * (num_candidate - actual_drop) / num_candidate
            aleatoric_cut = np.percentile(ua_mean, keep_percentage)
            candidate_mask = (ua_mean <= aleatoric_cut)
        else:
            candidate_mask = np.ones_like(ua_mean, dtype=bool)
            
        if (not np.any(candidate_mask)):
            candidate_mask = np.ones_like(ua_mean, dtype=bool)
        
        # The candidates MUST be within the remaining pool (we cannot re-pick uniform ones)
        valid_strategic_mask = candidate_mask & remaining_mask
        
        # Fallback if noise filtering was too aggressive and wiped out all remaining items
        if not np.any(valid_strategic_mask):
            valid_strategic_mask = remaining_mask

        #====================================================================================================#
        #====================================================================================================#
        # Get candidate indices and scores
        candidate_scores = scores[valid_strategic_mask]
        candidate_idx = np.where(valid_strategic_mask)[0]

        #====================================================================================================#
        #====================================================================================================#
        # Extract top performing candidates up to remaining strategic capacity
        top_candidate_sort_idx = np.argsort(candidate_scores)[::-1][:num_strategic]
        strategic_idx = candidate_idx[top_candidate_sort_idx]
        strategic_sample_indices = candidate_indices[strategic_idx]

        
        #====================================================================================================#
        #====================================================================================================#
        # Combine both selections
        return np.concatenate([uniform_sample_indices, strategic_sample_indices])

                
    def update_feature_dissimilarity(self, new_x, new_y, model):
        """
            Greedy K-Center on extracted features to maximize memory diversity
        """
        #print(f"\n\n==========> Dissimilarity-based memory update <==========\n\n")
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

    def update_hybrid_pool_kcenter(
        self,
        new_x,
        new_y,
        model,
        criterion,
        we=1.0,
        wH=0.1,
        wa=0.1,
        wl=1.0,
        alea_drop_fraction=0.15,
        mc_passes=10,
        uniform_ratio=0.5,
        pool_multiplier=3
    ):
        """
            Hybrid Idea 1: "Pool-Based K-Center".
            Retains X% uniformly. From the remaining samples, creates a pool of the highest scorers 
            (combining normalized loss and uncertainty). Finally, uses Greedy K-Center on the 
            latent features of that pool to maximize geometric diversity while ensuring informativeness.
        """
        # Safe MC Dropout (freeze batch norm, enable dropout layers)
        def enable_dropout(m):
            if type(m) == torch.nn.Dropout or type(m) == torch.nn.Dropout2d:
                m.train()
        model.eval()
        model.apply(enable_dropout)

        all_x = self.buffer_x + list(new_x.cpu())
        all_y = self.buffer_y + list(new_y.cpu())
        pool_size = len(all_x)
        target_capacity = min(self.capacity, pool_size)
        
        if (pool_size <= target_capacity):
            self.buffer_x = all_x
            self.buffer_y = all_y
            return

        # Batched computation of Uncertainties and Losses (to prevent GPU OOM)
        inputs = torch.stack(all_x).to(self.device)
        raw_epistemic, raw_aleatoric, raw_entropy = [], [], []
        all_losses = []

        with torch.no_grad():
            for i in range(0, len(inputs), 32): 
                batch_x = inputs[i:i+32]
                batch_y = torch.stack(all_y[i:i+32]).to(self.device)

                # ===> Loss Computation <===
                outputs = model(batch_x)
                # Ensure criterion outputs a per-sample loss (reduction='none' must be used)
                losses = criterion(outputs, batch_y).cpu()
                if losses.dim() == 0:
                    raise ValueError("Criterion must have reduction='none' for hybrid sampling.")
                all_losses.extend(losses.numpy())
                
                # ===> Uncertainty Computation <===
                probs = torch.stack([model(batch_x).softmax(dim=1) for _ in range(mc_passes)])
                
                entropies_per_pass = -torch.sum(probs * torch.log(probs + 1e-10), dim=2)
                aleatoric = entropies_per_pass.mean(dim=0)
                
                mean_probs = probs.mean(dim=0) 
                total_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1) 
                epistemic = total_entropy - aleatoric
                
                raw_epistemic.extend(epistemic.cpu().numpy())
                raw_aleatoric.extend(aleatoric.cpu().numpy())
                raw_entropy.extend(total_entropy.cpu().numpy())

        # Arrays for Normalization
        ua_mean = np.array(raw_aleatoric)
        ue_mean = np.array(raw_epistemic)
        h_mean = np.array(raw_entropy)
        l_mean = np.array(all_losses)

        # Min-Max Normalization Helper
        def min_max_normalize(arr):
            denominator = arr.max() - arr.min()
            if denominator == 0:
                return np.zeros_like(arr)
            return (arr - arr.min()) / denominator

        epistemic_norm = min_max_normalize(ue_mean)
        aleatoric_norm = min_max_normalize(ua_mean)
        entropy_norm = min_max_normalize(h_mean)
        loss_norm = min_max_normalize(l_mean)
        
        # Composite Score: Information maximization with noise mitigation
        scores = (we * epistemic_norm) + (wH * entropy_norm) + (wl * loss_norm) - (wa * aleatoric_norm)

        # Uniform Split Partitioning
        num_uniform = int(target_capacity * uniform_ratio)
        num_strategic = target_capacity - num_uniform
        all_indices = np.arange(pool_size)

        uniform_indices = np.random.choice(all_indices, num_uniform, replace=False)
        
        remaining_mask = np.ones(pool_size, dtype=bool)
        remaining_mask[uniform_indices] = False
        
        # Aleatoric Drop (Outlier Filtering)
        max_droppable = pool_size - target_capacity
        intended_drop = int(pool_size * alea_drop_fraction)
        actual_drop = min(intended_drop, max_droppable)
        
        if (actual_drop > 0):
            keep_percentage = 100.0 * (pool_size - actual_drop) / pool_size
            aleatoric_cut = np.percentile(ua_mean, keep_percentage)
            candidate_mask = (ua_mean <= aleatoric_cut)
        else:
            candidate_mask = np.ones_like(ua_mean, dtype=bool)
            
        if (not np.any(candidate_mask)):
            candidate_mask = np.ones_like(ua_mean, dtype=bool)
        
        valid_strategic_mask = candidate_mask & remaining_mask
        if not np.any(valid_strategic_mask):
            valid_strategic_mask = remaining_mask

        candidate_indices = np.where(valid_strategic_mask)[0]
        candidate_scores = scores[valid_strategic_mask]

        # Funnel into the Top Scoring Pool
        pool_target = min(len(candidate_indices), num_strategic * pool_multiplier)
        top_candidate_sort_idx = np.argsort(candidate_scores)[::-1][:pool_target]
        pool_indices = candidate_indices[top_candidate_sort_idx]

        # Geometric Diversity Check: Greedy K-Center on the selected pool
        if (len(pool_indices) <= num_strategic):
            # Skip K-center if the pool size is exactly what we need
            strategic_indices = pool_indices
        else:
            with torch.no_grad():
                pool_inputs = torch.stack([all_x[i] for i in pool_indices]).to(self.device)
                # Restore clean eval state before feature extraction
                model.eval() 
                pool_features = model.extract_features(pool_inputs).cpu()

            selected_pool_idx = [np.random.randint(0, len(pool_features))]
            min_distances = torch.norm(pool_features - pool_features[selected_pool_idx[0]], dim=1)

            # Iteratively pick the most geometrically distinct points from the high-scoring pool
            while len(selected_pool_idx) < num_strategic:
                farthest = torch.argmax(min_distances).item()
                selected_pool_idx.append(farthest)
                dist_to_new = torch.norm(pool_features - pool_features[farthest], dim=1)
                min_distances = torch.min(min_distances, dist_to_new)

            strategic_indices = [pool_indices[i] for i in selected_pool_idx]

        # Combine uniform and strategically-dissimilar arrays
        final_selected_indices = np.concatenate([uniform_indices, strategic_indices])

        self.buffer_x = [all_x[i] for i in final_selected_indices]
        self.buffer_y = [all_y[i] for i in final_selected_indices]


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
    def __init__(self, model, device, dataset_name=None):
        self.model = model.to(device) if model is not None else None
        self.device = device
        self.class_names = None
        if dataset_name is not None:
            if dataset_name.lower() == "hits":
                self.class_names = {idx: class_name for class_name, idx in HITSDataset.CLASS_TO_IDX.items()}
            elif dataset_name.lower() == "organmnist":
                if OrganMNISTHandler.IDX_TO_CLASS is None:
                    OrganMNISTHandler.load_class_mapping()
                self.class_names = OrganMNISTHandler.IDX_TO_CLASS
            elif dataset_name.lower() == "camelyon17":
                self.class_names = CamelyonHandler.IDX_TO_CLASS

    def extract_features_and_labels(self, dataloader):
        self.model.eval()
        features, labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    x, y = batch
                else:
                    x, y, *_ = batch

                if isinstance(y, tuple) or isinstance(y, list): y = y[0]
                feat = self.model.extract_features(x.to(self.device))
                features.append(feat.cpu().numpy())
                labels.append(y.cpu().numpy())
        return np.concatenate(features), np.concatenate(labels)

    def plot_memory_representation(self, task_a_loader, task_b_loader, memory_buffer, save_path=None):
        """visualize or save (if save_path provided) two figures, one showing the distribution of the two tasks and the memory buffer, and another showing the class distribution of the two tasks and the memory buffer.
        If the t-SNE dataset has already been computed and saved under save_path with "_dataset.csv" suffix, it will be loaded to avoid recomputation. In this case, provided loaders can be None.

        Args:
            task_a_loader (torch.utils.data.DataLoader): Dataloader of task A dataset.
            task_b_loader (torch.utils.data.DataLoader): Dataloader of task B dataset.
            memory_buffer (ReplayBuffer): Current memory buffer, samples can be redundant with loader A and B.
            save_path (str or Path, optional): Template path to save figures, for example path/to/Memory_0.png. Defaults to None.
        """
        print("\n\n==========> Extracting features for visualization <==========")

        # Getting save paths for dataset and plots
        save_path = Path(save_path) if save_path is not None else None
        dataset_save_path = None if save_path is None else save_path.with_name(f"{save_path.stem}_dataset.csv")
        task_plot_path = None if save_path is None else save_path.with_name(f"{save_path.stem}_tasks{save_path.suffix}")
        class_plot_path = None if save_path is None else save_path.with_name(f"{save_path.stem}_classes{save_path.suffix}")
       
        # Load  stored representation if exists 
        if dataset_save_path.exists():
            print(f"Dataset CSV already exists at {dataset_save_path}. Skipping t-SNE computation.")
            df = pd.read_csv(dataset_save_path)
            latent_2d = df[['tsne_1', 'tsne_2']].values
            labels_task = df['task_label'].values
            labels_A = df[df['task_label'] == 0]['class_label'].values
            labels_B = df[df['task_label'] == 1]['class_label'].values
            labels_mem = df[df['task_label'] == 2]['class_label'].values
        
        # Compute t-SNE projection
        else:
            # Extract whole datasets
            feat_A, labels_A = self.extract_features_and_labels(task_a_loader)
            feat_B, labels_B = self.extract_features_and_labels(task_b_loader)
            
            # Extract Memory Buffer
            mem_dataset = memory_buffer.get_dataset()
            if mem_dataset is None:
                print("Memory is empty. Cannot plot.")
                return
                
            mem_loader = DataLoader(mem_dataset, batch_size=32, shuffle=False)
            feat_mem, labels_mem = self.extract_features_and_labels(mem_loader)

            # Concatenate everything for t-SNE fitting
            all_features = np.vstack((feat_A, feat_B, feat_mem))
            
            # Generate Labels for coloring: 0 for Task A, 1 for Task B, 2 for Memory
            labels_task = np.array([0]*len(feat_A) + [1]*len(feat_B) + [2]*len(feat_mem))

            print("\n\n==========> Computing t-SNE (This may take a minute) <==========\n")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            latent_2d = tsne.fit_transform(all_features)

        # Split back for plotting
        latent_A = latent_2d[labels_task == 0]
        latent_B = latent_2d[labels_task == 1]
        latent_mem = latent_2d[labels_task == 2]
        
        #----- Save the dataset for external analysis
        if dataset_save_path is not None:
            df = pd.DataFrame({
                'tsne_1': latent_2d[:, 0],
                'tsne_2': latent_2d[:, 1],
                'task_label': labels_task,
                'class_label': np.concatenate((labels_A, labels_B, labels_mem))
            })
            df.to_csv(dataset_save_path, index=False)
            print(f"Saved t-SNE dataset to {dataset_save_path}")

        #----- Plot tasks and memory in a single figure
        task_colors = ['blue', 'green', 'red']  # Task A, Task B, Memory
        plt.figure(figsize=(10, 8))
        
        # Plot full datasets with low alpha (transparency)
        plt.scatter(latent_A[:, 0], latent_A[:, 1], c=task_colors[0], alpha=0.3, label='Task A Data', s=10)
        plt.scatter(latent_B[:, 0], latent_B[:, 1], c=task_colors[1], alpha=0.3, label='Task B Data', s=10)
        
        # Plot memory on top with distinct marker and high visibility
        plt.scatter(latent_mem[:, 0], latent_mem[:, 1], c=task_colors[2], marker='*', edgecolor='black', 
                    s=150, alpha=1.0, label='Memory Buffer')

        plt.title("t-SNE Projection: Dataset Distribution vs. Replay Memory")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend()
        plt.grid(True)
        if (save_path is None):
            plt.show()
        else:
            plt.savefig(task_plot_path, dpi=300)
        
        #----- Plot classes and memory in a single figure
        available_labels = list(np.unique(np.concatenate((labels_A, labels_B, labels_mem))))
        class_colors = {label: plt.cm.tab20(label) for label in available_labels}
        plt.figure(figsize=(10, 8))
    
        # Plot full datasets
        # Filled circles - task A
        plt.scatter(
            latent_A[:, 0],
            latent_A[:, 1],
            c=[class_colors[label] for label in labels_A],
            marker='o',
            edgecolors='none',
            alpha=0.8,
            s=20,
        )

        # Hollow circles - task B
        plt.scatter(
            latent_B[:, 0],
            latent_B[:, 1],
            facecolors='none',
            edgecolors=[class_colors[label] for label in labels_B],
            marker='o',
            linewidths=1.0,  # optional: adjust edge thickness
            alpha=0.3,
            s=20,
        )
        
        # Plot memory on top with high visibility
        plt.scatter(latent_mem[:, 0], latent_mem[:, 1], c=[class_colors[label] for label in labels_mem], marker='*', edgecolor='black', 
                    s=150, alpha=1.0, label='Memory Buffer')

        plt.title("t-SNE Projection: Class Distribution vs. Replay Memory")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        
        plt.legend(handles=
                        [plt.Line2D([0], [0], color="w",marker='o', label=f'Class {self.class_names[label]  if self.class_names is not None else label}', markerfacecolor=class_colors[label], markersize=10) for label in available_labels] + 
                        [plt.Line2D([0], [0], marker='*', color='w', label='Memory Buffer', markerfacecolor='white', markeredgecolor='black', markersize=15)] + 
                        [plt.Line2D([0], [0], marker='o', color="w", label='Task A', markerfacecolor='black', markersize=10)] + 
                        [plt.Line2D([0], [0], marker='o', color='w', label='Task B', markerfacecolor='white', markeredgecolor='black', markersize=10)],
                        loc='best')
        plt.grid(True)
        if (save_path is None):
            plt.show()
        else:
            plt.savefig(class_plot_path, dpi=300)

###-------------------- Visualizing memory representation --------------------###
if __name__ == "__main__":
    import argparse
    
    argparser = argparse.ArgumentParser(description="Test the LatentVisualizer with synthetic data.")
    argparser.add_argument("--save_path", type=str, default=None, help="Path to save the plots and dataset.")
    argparser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
    args = argparser.parse_args()
    
    # Plot memory representation testing
    latent_visualizer = LatentVisualizer(model=None, device='cpu', dataset_name=args.dataset_name)
    latent_visualizer.plot_memory_representation(
        task_a_loader=None,
        task_b_loader=None,
        memory_buffer=None,
        save_path=args.save_path # Path for which a dataset already exists
    )