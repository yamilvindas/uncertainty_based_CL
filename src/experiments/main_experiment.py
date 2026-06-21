#!/usr/bin/env python3
"""
    Main experiment to train and validate Continual Learning
    experiments for healthcare
"""
import os
import sys
import yaml
import argparse
from copy import deepcopy
import h5py
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import optuna
from optuna.trial import TrialState
from tqdm import tqdm

# For internal imports
sys.path.append(os.path.abspath(os.path.join("..")))
from src.data_processing.OrganMNIST import OrganMNISTHandler
from src.data_processing.Camelyon17 import CamelyonHandler
from src.continual_learning.memory import MemoryBuffer, LatentVisualizer
from src.continual_learning.ewc import EWC
from src.models.resnet import ResNet18CLModel
from src.models.simple_cnn import SimpleCLModel

class CLTrainer:
    def __init__(self, config):
        # Main config
        self.config = config

        # Device for computations
        self.device = torch.device(self.config.get("device", 'cuda:0'))

        # Define exp ID
        self.exp_id = self.config['exp_id']
        # Replay memory
        if (self.config['ContinualLearning']['Replay'].get('use_replay', False)):
            mem_strategy = self.config['ContinualLearning']['Replay'].get('memory_strategy', 'Uniform')
            mem_capacity = self.config['ContinualLearning']['Replay']['capacity']
            self.exp_id += f"_MemStrategy-{mem_strategy}_MemCapacity-{mem_capacity}"
        # EWC
        if (self.config['ContinualLearning']['EWC'].get('use_ewc', False)):
            self.exp_id += "_EWC-True"
        else:
            self.exp_id += "_EWC-False"
        
        # Setup Directories & Files
        self.base_results_dir = Path(self.config['results_dir'])
        self.res_dir = self.base_results_dir / self.exp_id
        self.models_dir = self.res_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.res_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.memories_dir = self.res_dir / "memories"
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        # Avoid overwriting previous files
        i = 0
        while (os.path.exists(self.metrics_dir / f"predictions_{i}.h5")):
            i += 1
        self.h5_path = self.metrics_dir / f"predictions_{i}.h5"
        
        # Initialize an empty HDF5 file
        with h5py.File(self.h5_path, 'w') as f:
            f.attrs['exp_id'] = self.exp_id

        # Initialize State Variables
        self.model = None
        self.memory = None
        self.memory_strategy = self.config['ContinualLearning']['Replay'].get('memory_strategy', 'Uniform')
        self.criterion = None 
        self.optimizer = None
        self.ewc = None
        self.previous_task_data_loader = None
        self.best_previous_task_model = None
        
        # Initial state setup based on default config
        self.reset_state(self.memory_strategy)

    def create_model(self):
        """
            Creates the model dynamically based on configuration.
        """
        # Get model type
        model_type = self.config['Model'].get('model_type', 'ResNet')

        # Create model
        if (model_type.lower() == 'resnet'):
            self.model = ResNet18CLModel(
                num_classes=self.config['Dataset']['num_classes'], 
                dropout_rate=self.config.get('dropout_rate', 0.5)
            ).to(self.device)

        elif (model_type.lower() in ['organcnn', 'camelyonmobilenet']):
            self.model = SimpleCLModel(
                                            model_type=model_type,
                                            num_classes=self.config['Dataset']['num_classes'], 
                                            dropout_rate=self.config.get('dropout_rate', 0.5)
                                        ).to(self.device)

        else:
            raise ValueError(f"Model type {model_type} is not valid.")
        
    def compute_class_weights(self, train_loader):
        """
            Extracts targets from the dataloader and computes balanced class weights.
        """
        print("Computing class weights for the current task...")
        all_targets = []
        
        with torch.no_grad():
            for batch in train_loader:
                # Get batch data
                if (self.config['Dataset'].get('dataset_type', 'OrganMNIST') == "Camelyon17"):
                    _, targets, _ = batch
                else:
                    _, targets = batch
                # Handle cases where targets might be wrapped in a list/tuple
                if (isinstance(targets, tuple) or isinstance(targets, list)):
                    targets = targets[0]
                all_targets.extend(targets.cpu().numpy())
                
        all_targets = np.array(all_targets)
        unique_classes = np.unique(all_targets)
        
        # Compute the balanced weights
        weights = compute_class_weight(
            class_weight='balanced', 
            classes=unique_classes, 
            y=all_targets
        )
    
        # Convert to a PyTorch tensor and move to the correct device
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        print(f"Computed Class Weights: {weight_tensor.cpu().numpy()}")
        return weight_tensor

    def reset_state(self, memory_strategy):
        """
            Wipes the model, memory, and EWC clean. Essential for Optuna trials.
        """
        # Reinitialize model
        self.create_model()

        # Redefine memory strategy
        self.memory_strategy = memory_strategy

        # Create empty memory
        if (self.config['ContinualLearning']['Replay'].get('use_replay', False)):
            self.memory = MemoryBuffer(self.config['ContinualLearning']['Replay']['capacity_in_n_samples'], self.device)
        else:
            self.memory = None
        
        # Remove EWC by default (if used it will be created from scratch later)
        self.ewc = None


    def models_have_same_weights(self, model1, model2):
        """
            Check if two models have the same parameters
        """
        # Check if they have the same number of parameters/keys
        if (model1.state_dict().keys() != model2.state_dict().keys()):
            return False
        
        # Compare the actual tensor values
        for key in model1.state_dict():
            if (not torch.equal(model1.state_dict()[key], model2.state_dict()[key])):
                return False
                
        return True

    def optimize_hyperparameters(self, task_data, n_trials=10):
        """
            Runs Optuna optimization, updates config, and prepares for the final run.
        """
        print(f"\n\n==========> Starting Optuna Optimization ({n_trials} trials) <==========")
        
        def objective(trial):
            # Suggest Hyperparameters
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
            if (self.config['ContinualLearning']['Replay'].get('use_replay', False)) and (self.current_task.lower() != 'task_a'):
                lambda_replay = trial.suggest_float("lambda_replay", 0.1, 5.0)
            else:
                lambda_replay = 0.0
            if (self.config['ContinualLearning']['EWC'].get('use_ewc', False)) and (self.current_task.lower() != 'task_a'):
                lambda_ewc = trial.suggest_float("lambda_ewc", 10.0, 15000.0)
            else:
                lambda_ewc = 0.0

            # For uncertainty-based sample selection
            # NOTE: we update the parameters of the memory selection strategie only if we are not in the first tast (memory is updated with the data loader of the previous task, so if we are in task B, we are going to use the data loader of task A)
            if (self.memory_strategy.lower() == 'uncertainty') and (self.current_task.lower() != "task_a"):
                # Suggest continuous weights between 0.0 and 2.0
                we = trial.suggest_float("we", 0.0, 2.0)
                wH = trial.suggest_float("wH", 0.0, 2.0)
                wa = trial.suggest_float("wa", 0.0, 2.0)
                alea_drop_fraction = trial.suggest_float("alea_drop_fraction", 0.0, 1.0)
                
                # Temporarily inject suggested params into the configuration state
                self.config['ContinualLearning']['Replay']['we'] = we
                self.config['ContinualLearning']['Replay']['wH'] = wH
                self.config['ContinualLearning']['Replay']['wa'] = wa
                self.config['ContinualLearning']['Replay']['alea_drop_fraction'] = alea_drop_fraction

                
            # Temporarily inject suggested params into the configuration state
            self.config['Training']['lr'] = lr
            self.config['Training']['weight_decay'] = weight_decay
            self.config['ContinualLearning']['Replay']['lambda_replay'] = lambda_replay
            self.config['ContinualLearning']['EWC']['lambda_ewc'] = lambda_ewc
            
            # Reset the environment for a completely isolated trial
            self.reset_state(self.memory_strategy)

            # Start optimization of current task with the best model of the previous task
            # NOTE: HAST TO BE DONE AFTER self.reset_state as self.reset_state resets the model too
            if (self.current_task.lower() != "task_a"):
                self.model = deepcopy(self.best_previous_task_model)

                # Re-initialize EWC for the Optuna trial
                if (self.config['ContinualLearning']['EWC'].get('use_ewc', False)):
                    previous_class_weights = self.compute_class_weights(self.previous_task_data_loader)
                    previous_criterion = nn.CrossEntropyLoss(weight=previous_class_weights.to(self.device), reduction='none')
                    self.ewc = EWC(self.model, self.previous_task_data_loader, self.device, previous_criterion)

            # Run train WITHOUT saving files to disk
            # Data loaders
            train_loader, val_loader, test_loader = self.get_data_loaders(task_data=task_data)
            eval_loaders = {
                                f'Val_{self.current_task}': val_loader,
                                f'Test_{self.current_task}': test_loader,
                            }
            # Class weights
            class_weights = self.compute_class_weights(train_loader)
            # Init
            self.initialize_task(class_weights)
            # Train
            self.train_single_task(self.current_task, train_loader, eval_loaders, rep=0, save_results=False)
            # Get final results after training on both tasks
            final_results = self.evaluate_and_save_phase(f"Phase_Post_{self.current_task}", eval_loaders, rep=0, save_results=False)
            val_metric_current_task = final_results[f'Val_{self.current_task}']
            # Save best model (it can be used for optimization of other tasks)
            self.save_model(self.current_task, rep=trial.number, optuna=True)
            
            # We want to maximize the current task metric
            return val_metric_current_task
        

        is_continual = (self.config['ContinualLearning']['Replay'].get('use_replay', False)) or (self.config['ContinualLearning']['EWC'].get('use_ewc', False))
        if (is_continual) and (self.current_task.lower() == 'task_a'):
            # In this case we do not optimiye the model for task A
            print(f"\n\n==========> IGNORING OPTUNA OPTIMIZATION FOR TASK A AS WE HAVE A CONTINUAL LEARNING EPXERIMENT (USING BASELINE OPTUNA RESULTS)\n\n")
            pass
        else:       
            # Define the SQLite database path inside the results folder
            db_path = os.path.join(self.metrics_dir, f"{self.exp_id}_{self.current_task}.db")
            storage_name = f"sqlite:///{db_path}"

            # Run Study
            study = optuna.create_study(
                                            direction="maximize", 
                                            study_name=f"{self.exp_id}_{self.current_task}",
                                            storage=storage_name,
                                            load_if_exists=True  # This allows resuming an interrupted study!
                                        )
            
            # Get number of completed trials in case we continue a study)
            completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            n_completed_trials = len(completed_trials)
            # Remaining number of trials to do
            n_remaining_trials_to_do = n_trials - n_completed_trials

            # Doing study only if target number of completed trials has not been reached (in case we continue a study)
            if (n_remaining_trials_to_do > 0):
                study.optimize(objective, n_trials=n_trials)

        # Save best model for current task
        if (is_continual) and (self.current_task.lower() == 'task_a'):
            dataset_type = self.config['Dataset'].get('dataset_type', 'OrganMNIST')
            # Get best trial number for baseline model
            # Study to load
            baseline_metrics_dir = self.base_results_dir / f"{dataset_type}_NoMemory_EWC-False" / "metrics" 
            db_path = os.path.join(baseline_metrics_dir, f"{dataset_type}_NoMemory_EWC-False_{self.current_task}.db")
            storage_name = f"sqlite:///{db_path}"
            other_study = optuna.load_study(study_name=f"{dataset_type}_NoMemory_EWC-False_{self.current_task}", storage=storage_name)
            best_trial_number = other_study.best_trial.number
            # We load the best model from the previous task from the Baseline
            model_path = self.base_results_dir / f"{dataset_type}_NoMemory_EWC-False" / "models" / f"model_{self.current_task}_rep-{best_trial_number}_optuna.pt"
            # TODO: IMPORTANT TO VERIFY IF THIS MAKE SENS FOR MORE THAN TWO TASKS
        else: # We are doing optimization in the baseline model
            # Best trial
            best_trial_number = study.best_trial.number
            # Model path
            model_path = self.models_dir / f"model_{self.current_task}_rep-{best_trial_number}_optuna.pt"
        self.load_model(model_path) # Load model
        self.best_previous_task_model = deepcopy(self.model)
        
        # Perform one final reset with the newly discovered optimal configuration
        self.reset_state(self.memory_strategy)
        print(f"\n\n==========> OTUNA optimization for task {self.current_task} finished <==========\n\n")


    def update_exp_params_optuna(self, task):
        """
            Updates the parameters of the experiment for the current task with the best found parameters with OPTUNA
        """
        is_continual = (self.config['ContinualLearning']['Replay'].get('use_replay', False)) or (self.config['ContinualLearning']['EWC'].get('use_ewc', False))
        if (is_continual) and (self.current_task.lower() == 'task_a'):
            # Get path to baseline trained model
            dataset_type = self.config['Dataset'].get('dataset_type', 'OrganMNIST')
            # Get storage
            baseline_metrics_dir = self.base_results_dir / f"{dataset_type}_NoMemory_EWC-False" / "metrics" 
            db_path = os.path.join(baseline_metrics_dir, f"{dataset_type}_NoMemory_EWC-False_{self.current_task}.db")
            storage_name = f"sqlite:///{db_path}"
            study = optuna.load_study(study_name=f"{dataset_type}_NoMemory_EWC-False_{self.current_task}", storage=storage_name)
        else:
            # Study to load
            db_path = os.path.join(self.metrics_dir, f"{self.exp_id}_{task}.db")
            storage_name = f"sqlite:///{db_path}"

            # Load study
            study = optuna.load_study(study_name=f"{self.exp_id}_{task}", storage=storage_name)

        # Retrieve and log best parameters
        best_params = study.best_params
        print(f"\n\n==========> Optuna Search Complet <==========")
        print(f"Best Trial Validation Score For Task {task}: {study.best_value:.4f}")
        print("Best Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print("\n\n")

        # Update class configuration with best parameters permanently
        self.config['Training']['lr'] = best_params['lr']
        self.config['Training']['weight_decay'] = best_params['weight_decay']
        if (self.config['ContinualLearning']['Replay'].get('use_replay', False)) and (self.current_task.lower() != 'task_a'):
            self.config['ContinualLearning']['Replay']['lambda_replay'] = best_params['lambda_replay']
        if (self.config['ContinualLearning']['EWC'].get('use_ewc', False)) and (self.current_task.lower() != 'task_a'):
            self.config['ContinualLearning']['EWC']['lambda_ewc'] = best_params['lambda_ewc']

        if (self.memory_strategy.lower() == 'uncertainty') and (task.lower() != "task_a"):
                # Temporarily inject suggested params into the configuration state
                self.config['ContinualLearning']['Replay']['we'] = best_params['we']
                self.config['ContinualLearning']['Replay']['wH'] = best_params['wH']
                self.config['ContinualLearning']['Replay']['wa'] = best_params['wa']
                self.config['ContinualLearning']['Replay']['alea_drop_fraction'] = best_params['alea_drop_fraction']

    def _save_to_h5(self, group_path, preds, targets, probs=None):
        """
            Helper to save arrays to HDF5 dynamically.
        """
        with h5py.File(self.h5_path, 'a') as f:
            group = f.require_group(group_path)
            if ('preds' in group):
                del group['preds']
            if ('targets' in group):
                del group['targets']
            if ('probs' in group): 
                del group['probs']
            group.create_dataset('preds', data=np.array(preds))
            group.create_dataset('targets', data=np.array(targets))
            if (probs is not None):
                group.create_dataset('probs', data=np.array(probs))


    def save_model(self, task_name, rep, optuna=False):
        """
            Saves the PyTorch model state with repetition tracking.
        """
        save_path = self.models_dir / f"model_{task_name}_rep-{rep}"
        if (optuna):
            save_path = self.models_dir / f"model_{task_name}_rep-{rep}_optuna.pt"
        else:
            save_path = self.models_dir / f"model_{task_name}_rep-{rep}.pt"
            
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, model_path):
        # Instantiate the model structure first
        self.create_model()
        
        # Load the state dictionary (using map_location to handle CPU/GPU routing gracefully)
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Load the weights into the model instance
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        
        print(f"==========> Model successfully loaded from {model_path} <==========")

    def initialize_task(self, class_weights):
        # Create loss function
        self.criterion = nn.CrossEntropyLoss(
                                                weight=class_weights.to(self.device),
                                                reduction='none'
                                            )
        # Create optimizer
        self.optimizer = optim.Adam(
                                        self.model.parameters(), 
                                        lr=self.config['Training']['lr'], 
                                        weight_decay=self.config['Training']['weight_decay']
                                    )

    def evaluate(self, test_loader):
        # Activate evaluation mode
        self.model.eval()
        # Get all the predictions
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                # Get batch data
                if (self.config['Dataset'].get('dataset_type', 'OrganMNIST') == "Camelyon17"):
                    x, y, metadata = batch
                else:
                    x, y = batch
                if (isinstance(y, tuple) or isinstance(y, list)):
                    y = y[0]

                outputs = self.model(x.to(self.device))
                probs = torch.softmax(outputs, dim=1) 
                preds = torch.argmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
                
        metric = balanced_accuracy_score(all_labels, all_preds)
        return metric, all_preds, all_labels, all_probs
    

    def update_memory(self, dataloader):
        if (self.memory is not None):
            for batch in dataloader:
                # Get batch data
                if (self.config['Dataset'].get('dataset_type', 'OrganMNIST') == "Camelyon17"):
                    x, y, metadata = batch
                else:
                    x, y = batch

                # Get input data and labels
                if (isinstance(y, tuple) or isinstance(y, list)):
                    y = y[0]
                x, y = x.to(self.device), y.to(self.device)

                # Update memory
                with torch.no_grad():
                    if (self.memory_strategy.lower() == 'uniform'):
                        self.memory.update_uniform(x, y)
                    elif (self.memory_strategy.lower() == 'loss'):
                        self.memory.update_loss_based(x, y, self.model, self.criterion)
                    elif (self.memory_strategy.lower() == 'uncertainty'):
                        self.memory.update_uncertainty_based(
                                                                new_x=x,
                                                                new_y=y, 
                                                                model=self.model,
                                                                we=self.config['ContinualLearning']['Replay']['we'],
                                                                wH=self.config['ContinualLearning']['Replay']['wH'],
                                                                wa=self.config['ContinualLearning']['Replay']['wa'],
                                                                alea_drop_fraction=self.config['ContinualLearning']['Replay']['alea_drop_fraction'],
                                                                mc_passes=self.config['ContinualLearning']['Replay']['mc_passes']
                                                            )
                    elif (self.memory_strategy.lower() == 'dissimilarity'):
                        self.memory.update_feature_dissimilarity(x, y, self.model)

    def train_single_task(self, task_name, train_loader, eval_loaders_dict, rep, save_results=True):
        # Update memory if necessary
        if (self.config['ContinualLearning']['Replay'].get('use_replay', False)) and (self.previous_task_data_loader is not None):
                self.update_memory(dataloader=self.previous_task_data_loader)

        # Activate train mode
        self.model.train()

        # Get some training parameters
        epochs = self.config['Training']['epochs']
        lambda_replay = self.config['ContinualLearning']['Replay'].get('lambda_replay', 1.0)

        # CL parameters
        lambda_replay = self.config['ContinualLearning']['Replay']['lambda_replay']
        lambda_ewc = self.config['ContinualLearning']['EWC']['lambda_ewc']

        # Get the memory dataset and dataloader
        if (self.memory is not None):
            mem_dataset = self.memory.get_dataset()
            mem_loader = DataLoader(mem_dataset, batch_size=train_loader.batch_size, shuffle=True) if mem_dataset else None
        else:
            mem_dataset = None
            mem_loader = None

        # Iterating over the dataset
        for epoch in tqdm(range(epochs)):
            self.model.train()
            if (mem_loader):
                mem_iter = iter(mem_loader)

            for batch in train_loader:
                # Get batch data
                if (self.config['Dataset'].get('dataset_type', 'OrganMNIST') == "Camelyon17"):
                    x, y, metadata = batch
                else:
                    x, y = batch

                # Get input data and labels
                if (isinstance(y, tuple) or isinstance(y, list)):
                    y = y[0]
                x, y = x.to(self.device), y.to(self.device)
                
                # Zeroing the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs_curr = self.model(x)

                # Loss computation
                total_loss = self.criterion(outputs_curr, y).mean()

                if (mem_loader):
                    try:
                        mem_x, mem_y = next(mem_iter)
                    except StopIteration:
                        mem_iter = iter(mem_loader)
                        mem_x, mem_y = next(mem_iter)
                        
                    # Forward pass on the memory only
                    outputs_mem = self.model(mem_x.to(self.device))

                    # Loss on the memory only
                    loss_mem = self.criterion(outputs_mem, mem_y.to(self.device)).mean()

                    # Add to total loss
                    total_loss += lambda_replay * loss_mem

                # Get EWC loss
                if (self.ewc is not None):
                    ewc_loss = self.ewc.penalty(self.model)
                    total_loss += (lambda_ewc / 2.0) * ewc_loss

                # Gradient computation
                total_loss.backward()

                # Parameters update
                self.optimizer.step()

            # Evaluate at the end of epoch
            for eval_name, loader in eval_loaders_dict.items():
                metric, preds, targets, probs = self.evaluate(loader)
                if save_results:
                    # Dynamically prepend the Repetition Index to the HDF5 group
                    h5_group = f"Rep_{rep}/{task_name}/epoch_{epoch}/{eval_name}"
                    self._save_to_h5(h5_group, preds, targets, probs)

                print(f"[{task_name}] Epoch {epoch+1}/{epochs} - {eval_name} Bal Acc.: {metric:.4f}")

    def evaluate_and_save_phase(self, phase_name, eval_loaders_dict, rep, save_results=True):
        # Results dict
        results = {}

        # Get the per-task results
        for eval_name, loader in eval_loaders_dict.items():
            metric, preds, targets, probs = self.evaluate(loader)
            if (save_results):
                self._save_to_h5(f"Rep_{rep}/{phase_name}/{eval_name}", preds, targets, probs)
            results[eval_name] = metric
        return results
    
    def get_data_loaders(self, task_data):
        # Data splitting
        train_data, val_data, test_data = task_data

        # Data loaders
        train_loader = DataLoader(train_data, batch_size=self.config['Training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config['Training']['batch_size'])
        test_loader = DataLoader(test_data, batch_size=self.config['Training']['batch_size'])

        return train_loader, val_loader, test_loader


    def repeated_holdout(self, task_a_data, task_b_data, ext_test_data=None, save_results=True, n_repetitions=5):
        # Get per-task data loaders
        self.loader_A, val_A_loader, test_A_loader = self.get_data_loaders(task_a_data)
        self.loader_B, val_B_loader, test_B_loader = self.get_data_loaders(task_b_data)
        
        # Define evaluation data loaders
        eval_loaders = {
                            'Val_Task_A': val_A_loader,
                            'Test_Task_A': test_A_loader,
                            'Val_Task_B': val_B_loader,
                            'Test_Task_B': test_B_loader
                        }

        # External validation set?
        if (ext_test_data):
            eval_loaders['External_Test'] = DataLoader(ext_test_data, batch_size=self.config['Training']['batch_size'])

        # Dictionary to aggregate results across multiple repetitions
        metrics_summary = {'Val_Task_A': [], 'Val_Task_B': [], 'Test_Task_A': [], 'Test_Task_B': []}
        if (ext_test_data):
            metrics_summary['Ext_Test'] = []

        for rep in range(n_repetitions):
            if (save_results):
                print(f"\n======================================")
                print(f"       STARTING REPETITION {rep + 1}/{n_repetitions}")
                print(f"======================================")
            # CRITICAL: Reset model, optimizers, EWC, and memory for every repetition
            self.reset_state(self.memory_strategy)

            # Save results BEFORE training on the tasks
            print("\n==========> Evaluating Before Training <==========\n")
            self.evaluate_and_save_phase("Phase_PreTraining", eval_loaders, rep, save_results)

            # --- Task A ---
            self.current_task = 'Task_A'
            print(f"\n==========> Training {self.current_task} <==========")
            self.previous_task = None
            self.previous_task_data_loader = None
            # Update hyper-parameters for current task with best OPTUNA hyper-parameters
            # NOTE: to do before self.initialize_task(class_weights)
            self.update_exp_params_optuna(task=self.current_task)
            # Class weights
            class_weights = self.compute_class_weights(self.loader_A)
            # Init
            self.initialize_task(class_weights)
            # Get trained model
            if (self.config['ContinualLearning']['Replay'].get('use_replay', False)) or (self.config['ContinualLearning']['EWC'].get('use_ewc', False)):
                # Get path to baseline trained model
                dataset_type = self.config['Dataset'].get('dataset_type', 'OrganMNIST')
                # Load the specific repetition model from the baseline, NOT the optuna model
                model_path = self.base_results_dir / f"{dataset_type}_NoMemory_EWC-False" / "models" / f"model_{self.current_task}_rep-{rep}.pt"
                # Load model
                self.load_model(model_path=model_path)
            else:
                # Train
                self.train_single_task(self.current_task, self.loader_A, eval_loaders, rep, save_results)
                # Save model
                if (save_results):
                    self.save_model(self.current_task, rep)
            self.evaluate_and_save_phase(f"Phase_Post_{self.current_task}", eval_loaders, rep, save_results)

            if (self.config['ContinualLearning']['EWC'].get('use_ewc', False)):
                print(f"\n\n==========> Computing Fisher Information Matrix for {self.current_task} <==========\n\n")
                self.ewc = EWC(self.model, self.loader_A, self.device, self.criterion)

            # --- Task B ---
            self.current_task = 'Task_B'
            print(f"\n==========> Training {self.current_task} <==========")
            self.previous_task = 'Task_A'
            # Update hyper-parameters for current task with best OPTUNA hyper-parameters
            # NOTE: to do before self.initialize_task(class_weights)
            self.update_exp_params_optuna(task=self.current_task)
            # Class weights
            class_weights = self.compute_class_weights(self.loader_B)
            # Init
            self.initialize_task(class_weights)
            # Train
            self.previous_task_data_loader = self.loader_A
            self.train_single_task(self.current_task, self.loader_B, eval_loaders, rep, save_results)
            # Save model
            if (save_results):
                self.save_model(self.current_task, rep)
                
            # Get final results after training on both tasks
            final_results = self.evaluate_and_save_phase(f"Phase_Post_{self.current_task}", eval_loaders, rep, save_results)
            # Save final results
            if (save_results):
                print(f"\n==========> Final Results Post-{self.current_task} <==========\n")
                for k, v in final_results.items():
                    print(f"{k}: {v:.4f}")

            # Store final results of this specific repetition
            for metric in metrics_summary.keys():
                metrics_summary[metric].append(final_results[metric])
                print(f"\t==========> Repetition {rep} {metric}: {final_results[metric]:.4f}")
                
            if (save_results):
                print(f"\n\n==========> End of Repetition {rep + 1} <==========\n\n")


            # Plot or save memory among all the training samples
            if (self.memory is not None):
                # File name
                memories_fig_path = self.memories_dir / f"Memory-{rep}.png"

                # Get computation device
                device = torch.device(self.config.get("device", 'cuda:0'))

                # Visualization
                visualizer = LatentVisualizer(self.model, device)
                visualizer.plot_memory_representation(self.loader_A, self.loader_B, self.memory, save_path=memories_fig_path)

        # Compute Mean and Standard Deviation
        final_means = {}
        if (save_results):
            print("\n======================================")
            print(f" FINAL AGGREGATE RESULTS ({n_repetitions} Runs)")
            print(f"======================================")
            
        for metric, values in metrics_summary.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            final_means[metric] = mean_val
            if save_results:
                print(f"\n\n==========> {metric}: {mean_val:.4f} ± {std_val:.4f}")
                
        return final_means['Val_Task_A'], final_means['Val_Task_B'], final_means['Test_Task_A'], final_means['Test_Task_B']


def set_seed(seed=42):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optional: set an environment variable for some libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"\n\n==========> Random seed set to: {seed} <==========")

    # To avoid some errors
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    #os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"


#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
def main():
    #====================================================================================================#
    #========================================= Argument Parsing =========================================#
    #====================================================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--parameters_file', required=True, help="Yaml parameters for the experiment", type=str)
    ap.add_argument('--seed', default=42, help="Seed to use for the experiment", type=int)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    parameters_file = args['parameters_file']
    with open(parameters_file, 'r') as file:
        config = yaml.safe_load(file)
    seed = args['seed']
    config['Seed'] = seed

    # Fix seed
    set_seed(seed)


    #====================================================================================================#
    #============================================ Experiment ============================================#
    #====================================================================================================#

    #====================================================================================================#
    # Setup data (using previous handlers)
    dataset_type = config['Dataset'].get('dataset_type', 'OrganMNIST')
    batch_size = config['Training'].get('batch_size', 64)
    if (dataset_type.lower() == "organmnist"):
        data_handler = OrganMNISTHandler(batch_size, lite=config['Dataset']['Lite'])
        task_a_data, task_b_data = data_handler.get_tasks()
        ext_test_data = None
    elif (dataset_type.lower() == "camelyon17"):
        data_handler = CamelyonHandler(batch_size=batch_size, lite=config['Dataset']['Lite'])
        task_a_data, task_b_data, ext_test_data = data_handler.get_tasks()
    else:
        raise ValueError(f"Dataset type {dataset_type} not valid.")
    
    #====================================================================================================#
    # Define number of possible samples in the memory 
    mem_capacity_samples = int(config['ContinualLearning']['Replay']['capacity']*data_handler.n_all_train_samples)
    config['ContinualLearning']['Replay']['capacity_in_n_samples'] = mem_capacity_samples
    print(f"\n\n==========> Memory capacity in number of samples: {mem_capacity_samples} (~{config['ContinualLearning']['Replay']['capacity']}%)")

    #====================================================================================================#
    # Initialize Trainer (Model is created internally based on YAML)
    trainer = CLTrainer(config=config)

    #====================================================================================================#
    # Run Hyperparameter Optimization 
    tasks = ['Task_A', 'Task_B']
    data_tasks = [task_a_data, task_b_data]
    loader_A, _, _ = trainer.get_data_loaders(task_a_data)
    loader_B, _, _ = trainer.get_data_loaders(task_b_data)
    train_loaders = [loader_A, loader_B]
    for i_task in range(len(tasks)):
        print(f"\n\n==========> Hyper-parameter optimization of task {tasks[i_task]} <==========\n\n")
        # Notice we pass the data. Optuna will test configurations and 
        # mutate the trainer's config to lock in the best parameters.
        current_task = tasks[i_task]
        trainer.current_task = current_task
        if (current_task.lower() != "task_a"): # We are not in the first task, so a previous data loader exists
            trainer.previous_task_data_loader = train_loaders[i_task-1]
            trainer.previous_task = tasks[i_task-1]
        else:
            trainer.previous_task_data_loader = None
            trainer.previous_task = None
            
        # Optimize
        trainer.optimize_hyperparameters(data_tasks[i_task], n_trials=config['Optuna'].get('n_trials', 10))


    #====================================================================================================#
    # Final Run (with saving enabled)
    # The trainer is currently loaded with the best hyperparameters
    # and a fresh model state.
    print("\n\n==========> Starting final full run with optimal configuration <==========\n")
    trainer.repeated_holdout(task_a_data, task_b_data, ext_test_data, save_results=True, n_repetitions=config['Training'].get('n_repetitions', 5))


if __name__ == "__main__":
    main()