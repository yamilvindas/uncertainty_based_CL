#!/usr/bin/env python3
"""
    Main experiment to train and validate Continual Learning
    experiments for healthcare
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader

from src.data_processing.Camelyon17 import CamelyonHandler
from src.data_processing.HITS import HITSHandler
from src.data_processing.OrganMNIST import OrganMNISTHandler
from src.experiments.main_experiment import CLTrainer, set_seed


class CLTester(CLTrainer):
    def __init__(self, config):
        super().__init__(config, inference_mode=True) # Read config, initialize repositories, initialize state variables and reset state
        self.training_dir = self.base_results_dir / self.exp_id
        self.models_list = []
        
        print("Inference directory:", self.res_dir)
        print("Training directory:", self.training_dir)

    def repeated_evaluation(self, task_a_data, task_b_data, ext_test_data=None):
        """
        Evaluate the model on the tasks (no training, just inference)
        """
        print(f"[INFO] Starting repeated evaluation for experiment: {self.exp_id}")
        n_repetitions = self.config['Training'].get('n_repetitions', 1)
        task_names = ['Task_A', 'Task_B']
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
        metrics_summary = {tn: {'Val_Task_A': [], 'Val_Task_B': [], 'Test_Task_A': [], 'Test_Task_B': []} for tn in task_names}
        if (ext_test_data):
            for tn in task_names:
                metrics_summary[tn]['External_Test'] = []

        for rep in range(n_repetitions):
            print(f"\n======================================")
            print(f"       STARTING REPETITION {rep + 1}/{n_repetitions}")
            print(f"======================================")

            # CRITICAL: Reset model, optimizers, EWC, and memory for every repetition
            self.reset_state(self.memory_strategy)
            
            # --- Task A ---
            self.current_task = task_names[0]
            print(f"\n==========> Evaluating {self.current_task} <==========")
            # Get path to baseline trained model correpsponding to the repetition
            dataset_type = self.config['Dataset'].get('dataset_type', 'OrganMNIST')
            model_path_A = self.base_results_dir / f"{dataset_type}_NoMemory_EWC-False" / "models" / f"model_{self.current_task}_rep-{rep}.pt"
            print(f"[INFO] Loading model from: {model_path_A}")
            # Load model
            self.load_model(model_path=model_path_A)
            # Evaluate and save results for Task A
            after_task_A_results = self.evaluate_and_save_phase(f"Phase_Post_{self.current_task}", eval_loaders, rep, save_results=True)

            # Store final results of this specific repetition
            for metric in metrics_summary[task_names[0]].keys():
                metrics_summary[task_names[0]][metric].append(after_task_A_results[metric])
                print(f"\t==========> Repetition {rep} {metric}: {after_task_A_results[metric]:.4f}")

            # --- Task B ---
            self.current_task = task_names[1]
            print(f"\n==========> Evaluating {self.current_task} <==========")
            # Get path to the experiment trained model correpsponding to the repetition
            model_path_B = self.training_dir / "models" / f"model_{self.current_task}_rep-{rep}.pt"
            print(f"[INFO] Loading model from: {model_path_B}")
            # Load model
            self.load_model(model_path=model_path_B)
            # Evaluate and save results for Task B
            final_results = self.evaluate_and_save_phase(f"Phase_Post_{self.current_task}", eval_loaders, rep, save_results=True)
            
            self.models_list.append((model_path_A, model_path_B))
        
            # Show final results
            print(f"\n==========> Final Results Post-{self.current_task} <==========\n")
            for k, v in final_results.items():
                print(f"{k}: {v:.4f}")
                
            # Store final results of this specific repetition
            for metric in metrics_summary[task_names[1]].keys():
                metrics_summary[task_names[1]][metric].append(final_results[metric])
                print(f"\t==========> Repetition {rep} {metric}: {final_results[metric]:.4f}")
            
            print(f"\n\n==========> End of Repetition {rep + 1} <==========\n\n")

        # Compute Mean and Standard Deviation
        final_means = {}
        final_stds = {}
        print("\n======================================")
        print(f" FINAL AGGREGATE RESULTS ({n_repetitions} Runs)")
        print(f"======================================")
        
        for task in task_names:
            print(f"\n==========> {task} <==========")
            final_means[task] = {}
            final_stds[task] = {}
            for metric, values in metrics_summary[task].items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                final_means[task][metric] = mean_val
                final_stds[task][metric] = std_val
                print(f"\n\n==========> {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Store evaluated model paths
        model_dict = {rep: {'Task_A': str(self.models_list[rep][0]), 'Task_B': str(self.models_list[rep][1])} for rep in range(n_repetitions)}
        with open(Path(self.metrics_dir) / "evaluated_models.json", 'w') as f:
            json.dump(model_dict, f, indent=4)
        
        return final_means, final_stds

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

    elif (dataset_type.lower() == "hits"):
        data_handler = HITSHandler(
                                        batch_size=batch_size,
                                        hdf5_a=config['Dataset']['task_a_hdf5'],
                                        hdf5_b=config['Dataset']['task_b_hdf5'],
                                        hdf5_ext=config['Dataset']['task_ext_hdf5'] if 'task_ext_hdf5' in config['Dataset'] else None,
                                    )
        task_a_data, task_b_data, ext_test_data = data_handler.get_tasks()

    else:
        raise ValueError(f"Dataset type {dataset_type} not valid.")

    #====================================================================================================#
    # Initialize Tester (Model is created internally based on YAML)
    tester = CLTester(config=config)

    #====================================================================================================#
    # DIRECTORY SETUP & INITIAL CONFIGURATION SAVE
    # Resolves to results/EXP_ID/configs/ grouping the config files with the experiment results
    configs_save_dir = Path(tester.metrics_dir).parent / "configs"
    configs_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create an isolated deepcopy of the configuration before Optuna suggestions run
    initial_config = deepcopy(tester.config)
    initial_config_path = configs_save_dir / "initial_config.yaml"
    with open(initial_config_path, 'w') as f:
        yaml.safe_dump(initial_config, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Saved initial configuration checkpoint to: {initial_config_path}")


    #====================================================================================================#
    # Evaluate the model on the tasks (no training, just inference) 

    # tester.repeated_holdout(task_a_data, task_b_data, ext_test_data, save_results=not(replay_memory_only), n_repetitions=config['Training'].get('n_repetitions', 5), replay_memory_only=replay_memory_only)
    final_means, final_stds = tester.repeated_evaluation(task_a_data, task_b_data, ext_test_data)
    
    #====================================================================================================#
    # Saving final means and stds
    with open(Path(tester.metrics_dir) / "final_results.json", 'w') as f:
        json.dump({'means': final_means, 'stds': final_stds}, f, indent=4)

if __name__ == "__main__":
    main()