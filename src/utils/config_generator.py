"""
    Generates the config files to run the different experiments.
"""
import os
import yaml

def build_base_config(dataset_name):
    """
        Returns the core/default hyperparameters for each dataset.
    """
    if (dataset_name == "Camelyon17"):
        return {
                    "exp_id": "Camelyon17",
                    "device": "cuda:0",
                    "results_dir": "./results",
                    "ContinualLearning": {
                                            "Replay": {
                                                            "use_replay": False,
                                                            "memory_strategy": "Uniform",
                                                            "capacity": 0.01,
                                                            "lambda_replay": 1.0
                                                        },
                                            "EWC": {
                                                        "use_ewc": False,
                                                        "lambda_ewc": 1.0e3
                                                    }
                                        },
                    "Dataset": {
                        "dataset_type": "Camelyon17",
                        "Lite": True,
                        "num_classes": 2
                    },

                    "Model": {
                                "model_type": "CamelyonMobileNet"
                             },

                    "Training": {
                                    "epochs": 20,
                                    "lr": 1.0e-3,
                                    "weight_decay": 1.0e-5,
                                    "batch_size": 64,
                                    "n_repetitions": 10
                                },

                    "Optuna": {
                                "n_trials": 30
                            }
                }
    
    elif (dataset_name == "OrganMNIST"):
        return {
                    "exp_id": "OrganMNIST",
                    "device": "cuda:0",
                    "results_dir": "./results",
                    "ContinualLearning": {
                                            "Replay": {
                                                            "use_replay": True,
                                                            "memory_strategy": "Uniform",
                                                            "capacity": 0.1,
                                                            "lambda_replay": 1.0
                                                        },
                                            "EWC": {
                                                        "use_ewc": False,
                                                        "lambda_ewc": 1.0e3
                                                    }
                    },

                    "Dataset": {
                                    "dataset_type": "OrganMNIST",
                                    "Lite": True,
                                    "num_classes": 11
                                },

                    "Model": {
                                "model_type": "OrganCNN"
                            },

                    "Training": {
                                    "epochs": 20,
                                    "lr": 1.0e-3,
                                    "weight_decay": 1.0e-5,
                                    "batch_size": 64,
                                    "n_repetitions": 10
                                },

                    "Optuna": {
                                    "n_trials": 30
                                }
                }
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def write_yaml(filepath, data):
    """
        Safely writes data to a YAML file, ensuring directories are created.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Generated: {filepath}")

def generate_all_configs():
    # Base directories
    base_dir = "configs"

    # Datasets
    datasets = ["Camelyon17", "OrganMNIST"]
    
    # Memory configurations with their respective capacity ratios
    mem_configs = {
                        "Mem-1": 0.01,
                        "Mem-5": 0.05,
                        "Mem-10": 0.10,
                        "Mem-50": 0.50,
                        "Mem-100": 1.00
                    }
    
    for dataset in datasets:
        # Create Baseline Configs
        baseline_path = os.path.join(base_dir, dataset, "Baseline")
        
        # ===> No Memory Baseline <===
        no_mem_config = build_base_config(dataset)
        no_mem_config["exp_id"] = f"{dataset}_NoMemory"
        no_mem_config["ContinualLearning"]["Replay"]["use_replay"] = False
        no_mem_config["ContinualLearning"]["Replay"]["capacity"] = 0.0
        no_mem_config["ContinualLearning"]["Replay"]["lambda_replay"] = 0.0
        no_mem_config["ContinualLearning"]["EWC"]["use_ewc"] = False
        no_mem_config["ContinualLearning"]["EWC"]["lambda_ewc"] = 0.0
        
        write_yaml(os.path.join(baseline_path, "NoMemory.yaml"), no_mem_config)
        
        # ===> EWC Only Baseline <===
        ewc_config = build_base_config(dataset)
        ewc_config["exp_id"] = f"{dataset}_EWC"
        ewc_config["ContinualLearning"]["Replay"]["use_replay"] = False
        ewc_config["ContinualLearning"]["Replay"]["capacity"] = 0.0
        ewc_config["ContinualLearning"]["Replay"]["lambda_replay"] = 0.0
        ewc_config["ContinualLearning"]["EWC"]["use_ewc"] = True
        ewc_config["ContinualLearning"]["EWC"]["lambda_ewc"] = 1.0e3
        
        write_yaml(os.path.join(baseline_path, "EWC.yaml"), ewc_config)
        
        # Create Replay and Replay+EWC Configs for each capacity ratio
        for mem_folder, capacity_ratio in mem_configs.items():
            # Change folder path
            folder_path = os.path.join(base_dir, dataset, mem_folder)

            # Possible strategies
            mem_select_strategies = ['uniform', 'uncertainty', 'loss', 'dissimilarity']
            for mem_strategy in mem_select_strategies:
                # ===> Replay Only <===
                replay_only = build_base_config(dataset)
                replay_only["exp_id"] = f"{dataset}_{mem_folder}_Replay"
                replay_only["ContinualLearning"]["Replay"]["use_replay"] = True
                replay_only["ContinualLearning"]["Replay"]["capacity"] = capacity_ratio
                replay_only["ContinualLearning"]["Replay"]["memory_strategy"] = mem_strategy
                replay_only["ContinualLearning"]["Replay"]["lambda_replay"] = 1.0
                replay_only["ContinualLearning"]["EWC"]["use_ewc"] = False
                replay_only["ContinualLearning"]["EWC"]["lambda_ewc"] = 0.0
                if (mem_strategy.lower() == 'uncertainty'):
                    replay_only['ContinualLearning']['Replay']['we'] = 1.0
                    replay_only['ContinualLearning']['Replay']['wH'] = 1.0
                    replay_only['ContinualLearning']['Replay']['wa'] = 1.0
                    replay_only['ContinualLearning']['Replay']['alea_drop_fraction'] = 0.15
                    replay_only['ContinualLearning']['Replay']['mc_passes'] = 10
                
                write_yaml(os.path.join(folder_path, f"Replay-{mem_strategy}.yaml"), replay_only)
                
                # ===> Replay with EWC <===
                replay_ewc = build_base_config(dataset)
                replay_ewc["exp_id"] = f"{dataset}_{mem_folder}_ReplayEWC"
                replay_ewc["ContinualLearning"]["Replay"]["use_replay"] = True
                replay_ewc["ContinualLearning"]["Replay"]["capacity"] = capacity_ratio
                replay_ewc["ContinualLearning"]["Replay"]["lambda_replay"] = 1.0
                replay_ewc["ContinualLearning"]["Replay"]["memory_strategy"] = mem_strategy
                replay_ewc["ContinualLearning"]["EWC"]["use_ewc"] = True
                replay_ewc["ContinualLearning"]["EWC"]["lambda_ewc"] = 1.0e3
                if (mem_strategy.lower() == 'uncertainty'):
                    replay_ewc['ContinualLearning']['Replay']['we'] = 1.0
                    replay_ewc['ContinualLearning']['Replay']['wH'] = 1.0
                    replay_ewc['ContinualLearning']['Replay']['wa'] = 1.0
                    replay_ewc['ContinualLearning']['Replay']['alea_drop_fraction'] = 0.15
                    replay_ewc['ContinualLearning']['Replay']['mc_passes'] = 10
                
                write_yaml(os.path.join(folder_path, f"Replay-{mem_strategy}_EWC.yaml"), replay_ewc)

if __name__ == "__main__":
    generate_all_configs()
    print("\n\n=======> All configuration folders and files have been successfully generated! <=======\n\n")