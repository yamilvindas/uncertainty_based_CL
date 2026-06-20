#!/usr/bin/env python3
"""
    Main experiment to train and validate Continual Learning
    experiments for healthcare
"""
import os
import sys
import yaml
import argparse
import h5py
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score
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
        self.res_dir = Path(self.config['results_dir']) / self.exp_id
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

    def optimize_hyperparameters(self, task_a_data, task_b_data, n_trials=10):
        """
            Runs Optuna optimization, updates config, and prepares for the final run.
        """
        print(f"\n\n==========> Starting Optuna Optimization ({n_trials} trials) <==========")
        
        def objective(trial):
            # Suggest Hyperparameters
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
            if (self.config['ContinualLearning']['Replay'].get('use_replay', False)):
                lambda_replay = trial.suggest_float("lambda_replay", 0.1, 5.0)
            else:
                lambda_replay = 0.0
            if (self.config['ContinualLearning']['EWC'].get('use_ewc', False)):
                lambda_ewc = trial.suggest_float("lambda_ewc", 10.0, 15000.0)
            else:
                lambda_ewc = 0.0

            # For uncertainty-based sample selection
            if (self.memory_strategy.lower() == 'uncertainty'):
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

            # Unpack the 4 returned evaluation results from repeated_holdout
            # Run holdout WITHOUT saving files to disk
            val_metric_a, val_metric_b, test_metric_a, test_metric_b = self.repeated_holdout(task_a_data, task_b_data, save_results=False, n_repetitions=1)
            
            # We want to maximize the average metric of both tasks
            # TODO: OPTIMIZE FORGETTING???
            return (val_metric_a + val_metric_b) / 2.0
        
        # Define the SQLite database path inside the results folder
        db_path = os.path.join(self.metrics_dir, f"{self.exp_id}.db")
        storage_name = f"sqlite:///{db_path}"

        # Run Study
        study = optuna.create_study(
                                        direction="maximize", 
                                        study_name=self.exp_id,
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

        # Retrieve and log best parameters
        best_params = study.best_params
        print(f"\n\n==========> Optuna Search Complet <==========")
        print(f"Best Trial Validation Score: {study.best_value:.4f}")
        print("Best Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print("\n\n")
            
        # Update class configuration with best parameters permanently
        self.config['Training']['lr'] = best_params['lr']
        self.config['Training']['weight_decay'] = best_params['weight_decay']
        if (self.config['ContinualLearning']['Replay'].get('use_replay', False)):
            self.config['ContinualLearning']['Replay']['lambda_replay'] = best_params['lambda_replay']
        if (self.config['ContinualLearning']['EWC'].get('use_ewc', False)):
            self.config['ContinualLearning']['EWC']['lambda_ewc'] = best_params['lambda_ewc']
        
        # Perform one final reset with the newly discovered optimal configuration
        self.reset_state(self.memory_strategy)
        print("\n\n==========> Trainer state reset with optimal hyperparameters. Ready for final run. <==========\n\n")

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

    def save_model(self, task_name):
        save_path = self.models_dir / f"model_{task_name}.pt"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def save_model(self, task_name, rep):
        """
            Saves the PyTorch model state with repetition tracking.
        """
        save_path = self.models_dir / f"model_{task_name}_rep{rep}.pt"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

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

    def train_single_task(self, task_name, train_loader, eval_loaders_dict, rep, save_results=True):
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

                # Update memory
                if (self.memory is not None):
                    # NOTE: we do it at EACH epoch as there some methods (like loss-based or uncertainty-based) are model dependent so some samples can become useless or useful over the epochs
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


    def repeated_holdout(self, task_a_data, task_b_data, ext_test_data=None, save_results=True, n_repetitions=5):
        # Get per-task datasets
        train_A, val_A, test_A = task_a_data
        train_B, val_B, test_B = task_b_data
        
        # Get per-task dataloaders
        self.loader_A = DataLoader(train_A, batch_size=self.config['Training']['batch_size'], shuffle=True)
        self.loader_B = DataLoader(train_B, batch_size=self.config['Training']['batch_size'], shuffle=True)
        
        # Define evaluation data loaders
        eval_loaders = {
                            'Val_A': DataLoader(val_A, batch_size=self.config['Training']['batch_size']),
                            'Test_A': DataLoader(test_A, batch_size=self.config['Training']['batch_size']),
                            'Val_B': DataLoader(val_B, batch_size=self.config['Training']['batch_size']),
                            'Test_B': DataLoader(test_B, batch_size=self.config['Training']['batch_size'])
                        }

        # External validation set?
        if (ext_test_data):
            eval_loaders['External_Test'] = DataLoader(ext_test_data, batch_size=self.config['Training']['batch_size'])

        # Dummy class weights
        dummy_weights = torch.ones(self.config['Dataset']['num_classes']) 

        # Dictionary to aggregate results across multiple repetitions
        metrics_summary = {'Val_A': [], 'Val_B': [], 'Test_A': [], 'Test_B': []}
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
            if (save_results):
                print("\n==========> Evaluating Before Training <==========\n")
            self.evaluate_and_save_phase("Phase_PreTraining", eval_loaders, rep, save_results)

            # --- Task A ---
            if (save_results):
                print("\n==========> Training Task A <==========")
            # Init
            self.initialize_task(dummy_weights)
            # Train
            self.train_single_task("Task_A", self.loader_A, eval_loaders, rep, save_results)
            # Save model
            if (save_results):
                self.save_model("Task_A", rep)
            self.evaluate_and_save_phase("Phase_Post_Task_A", eval_loaders, rep, save_results)

            if (self.config.get('use_ewc', False)):
                if (save_results):
                    print("\n\n==========> Computing Fisher Information Matrix for Task A <==========\n\n")
                self.ewc = EWC(self.model, self.loader_A, self.device, self.criterion)

            # --- Task B ---
            if (save_results):
                print("\n==========> Training Task B <==========")
            # Init
            self.initialize_task(dummy_weights)
            # Train
            self.train_single_task("Task_B", self.loader_B, eval_loaders, rep, save_results)
            # Save model
            if (save_results):
                self.save_model("Task_B", rep)

                
            # Get final results after training on both tasks
            final_results = self.evaluate_and_save_phase("Phase_Post_Task_B", eval_loaders, rep, save_results)
            # Save final results
            if (save_results):
                print("\n==========> Final Results Post-Task B <==========\n")
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
                
        return final_means['Val_A'], final_means['Val_B'], final_means['Test_A'], final_means['Test_B']


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
    
    # Define number of possible samples in the memory 
    mem_capacity_samples = int(config['ContinualLearning']['Replay']['capacity']*data_handler.n_all_train_samples)
    config['ContinualLearning']['Replay']['capacity_in_n_samples'] = mem_capacity_samples
    print(f"\n\n==========> Memory capacity in number of samples: {mem_capacity_samples} (~{config['ContinualLearning']['Replay']['capacity']}%)")

    # Initialize Trainer (Model is created internally based on YAML)
    trainer = CLTrainer(config=config)

    # Run Hyperparameter Optimization 
    # Notice we pass the data. Optuna will test configurations and 
    # mutate the trainer's config to lock in the best parameters.
    trainer.optimize_hyperparameters(task_a_data, task_b_data, n_trials=config['Optuna'].get('n_trials', 10))

    # Final Run (with saving enabled)
    # The trainer is currently loaded with the best hyperparameters
    # and a fresh model state.
    print("\n\n==========> Starting final full run with optimal configuration <==========\n")
    trainer.repeated_holdout(task_a_data, task_b_data, ext_test_data, save_results=True, n_repetitions=config['Training'].get('n_repetitions', 5))


if __name__ == "__main__":
    main()