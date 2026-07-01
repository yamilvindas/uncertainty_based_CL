#!/usr/bin/env python3
"""
    Class defined the memory buffer for replay-based learning.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

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
        if (save_path is None):
            plt.show()
        else:
            plt.savefig(save_path, dpi=300)

