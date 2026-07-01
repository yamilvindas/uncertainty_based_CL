#!/usr/bin/env python3
"""
    Main experiment to train and validate Continual Learning
    experiments for healthcare
"""
import copy

import torch


class EWC:
    def __init__(self, model, dataloader, device, criterion):
        # Main attributes
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.criterion = criterion
        
        # Save optimal parameters of the previous task
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._saved_params = {n: p.clone().detach() for n, p in self.params.items()}
        
        # Compute Fisher Information Matrix
        self._precision_matrices = self._compute_fisher()

    def _compute_fisher(self):
        """
            Computes the Fisher information Matrix
        """
        # Initialize the matrix for each parameter
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p.data)

        # Compute the matrix with the given dataloader
        self.model.eval()
        for batch in self.dataloader:
            if len(batch) == 2:
                x, y = batch
            else:
                x, y, *_ = batch
            # Get input data and labels
            if (isinstance(y, tuple) or isinstance(y, list)):
                y = y[0]
            x, y = x.to(self.device), y.to(self.device)
            
            # Zeroing gradients
            self.model.zero_grad()

            # Get output of the model
            outputs = self.model(x)
            
            # Using log-likelihood for Fisher (CrossEntropy is negative log-likelihood)
            loss = self.criterion(outputs, y).mean()
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    # Square of gradients serves as an empirical estimate of Fisher Information
                    precision_matrices[n].data += (p.grad.data ** 2) / len(self.dataloader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        """
            Calculates the EWC regularization loss.
        """
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._precision_matrices:
                _loss = self._precision_matrices[n] * (p - self._saved_params[n]) ** 2
                loss += _loss.sum()
        return loss