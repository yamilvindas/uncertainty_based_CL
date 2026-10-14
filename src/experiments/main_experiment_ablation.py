#!/usr/bin/env python3
"""
    main_experiment_ablation.py

    Runner for the component-wise ablation study (we / wH / wa) of the
    Proposed Model (Uncertainty Replay + ByClass, no EWC).

    Does NOT modify main_experiment.py. Subclasses CLTrainer and overrides
    ONLY optimize_hyperparameters() and update_exp_params_optuna() -- both
    method bodies below are copied verbatim from main_experiment.py, with
    minimal, clearly-marked changes: the ablated component's weight is
    fixed to 0.0 and REMOVED from the Optuna search space entirely (not
    just zeroed after the fact), instead of being suggested by
    trial.suggest_float(...) like the other two weights. Every other line
    (lr, weight_decay, lambda_replay, lambda_ewc, the other two weights,
    alea_drop_fraction, uniform_ratio, reset/train/eval logic, model
    saving, study creation/resuming) is unchanged.

    Usage (identical to main_experiment.py, just a different entry point):
        python src/experiments/main_experiment_ablation.py \\
            --parameters_file configs/HITS/Ablation/Mem-10/Ablation-xwe.yaml
"""
import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import optuna
import torch.nn as nn
import yaml
from optuna.trial import TrialState

sys.path.append(os.path.abspath(os.path.join("..")))
from src.continual_learning.ewc import EWC
from src.data_processing.Camelyon17 import CamelyonHandler
from src.data_processing.HITS import HITSHandler
from src.data_processing.OrganMNIST import OrganMNISTHandler

# Reuse everything from the original, unmodified file
from main_experiment import CLTrainer, set_seed


class AblationCLTrainer(CLTrainer):
    def __init__(self, config, inference_mode=False):
        super().__init__(config, inference_mode=inference_mode)
        self.ablate_component = self.config.get("Ablation", {}).get("ablate_component")
        if self.ablate_component not in ("we", "wH", "wa"):
            raise ValueError(
                f"Ablation config must set Ablation.ablate_component to one of "
                f"'we', 'wH', 'wa' -- got: {self.ablate_component!r}"
            )
        print(f"\n\n==========> ABLATION MODE: '{self.ablate_component}' is fixed to 0.0 "
              f"and excluded from the Optuna search space <==========\n\n")

    def optimize_hyperparameters(self, task_data, n_trials=10):
        """
            Identical to CLTrainer.optimize_hyperparameters(), except the
            ablated component (self.ablate_component) is fixed to 0.0
            instead of being suggested by Optuna. Every other line is
            unchanged from main_experiment.py.
        """
        if (not self.use_optuna):
            print("\n\n==========> Skipping Optuna optimization; using configuration parameters as-is <==========\n")
            return

        print(f"\n\n==========> Starting Optuna Optimization ({n_trials} trials) <==========")

        def objective(trial):
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

            if (self.memory_strategy.lower() in ['uncertainty', 'hybrid']) and (self.current_task.lower() != "task_a"):
                # ---- ABLATION CHANGE: the ablated component is fixed to
                # 0.0 and is NOT suggested by Optuna (removed from the
                # search space entirely) -- the other two weights are
                # searched exactly as in the original method. ----
                we = 0.0 if self.ablate_component == "we" else trial.suggest_float("we", 0.0, 2.0)
                wH = 0.0 if self.ablate_component == "wH" else trial.suggest_float("wH", 0.0, 2.0)
                wa = 0.0 if self.ablate_component == "wa" else trial.suggest_float("wa", 0.0, 2.0)
                alea_drop_fraction = trial.suggest_float("alea_drop_fraction", 0.0, 0.5)

                self.config['ContinualLearning']['Replay']['we'] = we
                self.config['ContinualLearning']['Replay']['wH'] = wH
                self.config['ContinualLearning']['Replay']['wa'] = wa
                self.config['ContinualLearning']['Replay']['alea_drop_fraction'] = alea_drop_fraction

                # Kept for structural parity with the original method;
                # unused for the Proposed Model, which is not Hybrid.
                if (self.memory_strategy.lower() == 'hybrid'):
                    wl = trial.suggest_float("wl", 0.0, 2.0)
                    pool_multiplier = trial.suggest_int("pool_multiplier", 2, 8)
                    self.config['ContinualLearning']['Replay']['wl'] = wl
                    self.config['ContinualLearning']['Replay']['pool_multiplier'] = pool_multiplier

            if (self.memory_strategy.lower() in ['uncertainty', 'loss', 'hybrid']) and (self.current_task.lower() != "task_a"):
                if (self.config['ContinualLearning']['Replay'].get('optimize_uniform_ratio', True)):
                    uniform_ratio = trial.suggest_float("uniform_ratio", 0.0, 1.0)
                    self.config['ContinualLearning']['Replay']['uniform_ratio'] = uniform_ratio

            self.config['Training']['lr'] = lr
            self.config['Training']['weight_decay'] = weight_decay
            self.config['ContinualLearning']['Replay']['lambda_replay'] = lambda_replay
            self.config['ContinualLearning']['EWC']['lambda_ewc'] = lambda_ewc

            self.reset_state(self.memory_strategy)

            if (self.current_task.lower() != "task_a"):
                self.model = deepcopy(self.best_previous_task_model)

                if (self.config['ContinualLearning']['EWC'].get('use_ewc', False)):
                    print("Computing class weights for the previous task to initialize EWC...")
                    previous_class_weights, _ = self.compute_class_weights(self.previous_task_data_loader['Train'])
                    previous_criterion = nn.CrossEntropyLoss(weight=previous_class_weights.to(self.device), reduction='none')
                    self.ewc = EWC(self.model, self.previous_task_data_loader['Train'], self.device, previous_criterion)

            train_loader, val_loader, test_loader = self.get_data_loaders(task_data=task_data)
            eval_loaders = {
                f'Val_{self.current_task}': val_loader,
                f'Test_{self.current_task}': test_loader,
            }
            if (self.previous_task_data_loader is not None):
                eval_loaders.update(
                    {
                        f'Val_{self.previous_task}': self.previous_task_data_loader['Val'],
                        f'Test_{self.previous_task}': self.previous_task_data_loader['Test'],
                    }
                )
            print(f"Computing class weights for Task {self.current_task}...")
            class_weights, _ = self.compute_class_weights(train_loader)
            self.initialize_task(class_weights)
            self.train_single_task(self.current_task, train_loader, eval_loaders, rep=0, save_results=False)
            final_results = self.evaluate_and_save_phase(f"Phase_Post_{self.current_task}", eval_loaders, rep=0, save_results=False)
            val_metric_current_task = final_results[f'Val_{self.current_task}']
            if (self.current_task.lower() != 'task_a'):
                val_metric_previous_task = final_results[f'Val_{self.previous_task}']
                val_metric = (val_metric_current_task + val_metric_previous_task) / 2
            else:
                val_metric = val_metric_current_task
            self.save_model(self.current_task, rep=trial.number, optuna=True)

            return val_metric

        is_continual = (self.config['ContinualLearning']['Replay'].get('use_replay', False)) or (self.config['ContinualLearning']['EWC'].get('use_ewc', False))
        if (is_continual) and (self.current_task.lower() == 'task_a'):
            print(f"\n\n==========> IGNORING OPTUNA OPTIMIZATION FOR TASK A AS WE HAVE A CONTINUAL LEARNING EXPERIMENT (USING BASELINE OPTUNA RESULTS)\n\n")
            pass
        else:
            db_path = os.path.join(self.metrics_dir, f"{self.exp_id}_{self.current_task}.db")
            storage_name = f"sqlite:///{db_path}"

            study = optuna.create_study(
                direction="maximize",
                study_name=f"{self.exp_id}_{self.current_task}",
                storage=storage_name,
                sampler=optuna.samplers.TPESampler(seed=self.config.get('Seed', 42)),
                load_if_exists=True
            )

            completed_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            n_completed_trials = len(completed_trials)
            n_remaining_trials_to_do = n_trials - n_completed_trials

            if (n_remaining_trials_to_do > 0):
                study.optimize(objective, n_trials=n_remaining_trials_to_do)

        if (is_continual) and (self.current_task.lower() == 'task_a'):
            dataset_type = self.config['Dataset'].get('dataset_type', 'OrganMNIST')
            baseline_metrics_dir = self.base_results_dir / f"{dataset_type}_NoMemory_EWC-False" / "metrics"
            db_path = os.path.join(baseline_metrics_dir, f"{dataset_type}_NoMemory_EWC-False_{self.current_task}.db")
            storage_name = f"sqlite:///{db_path}"
            other_study = optuna.load_study(study_name=f"{dataset_type}_NoMemory_EWC-False_{self.current_task}", storage=storage_name)
            best_trial_number = other_study.best_trial.number
            model_path = self.base_results_dir / f"{dataset_type}_NoMemory_EWC-False" / "models" / f"model_{self.current_task}_rep-{best_trial_number}_optuna.pt"
        else:
            best_trial_number = study.best_trial.number
            model_path = self.models_dir / f"model_{self.current_task}_rep-{best_trial_number}_optuna.pt"
        self.load_model(model_path)
        self.best_previous_task_model = deepcopy(self.model)

        self.reset_state(self.memory_strategy)
        print(f"\n\n==========> OPTUNA optimization for task {self.current_task} finished <==========\n\n")

    def update_exp_params_optuna(self, task):
        """
            Identical to CLTrainer.update_exp_params_optuna(), except it
            uses best_params.get(x, 0.0) instead of best_params[x] for the
            ablated component, since that component was never part of the
            Optuna search space and will genuinely be absent from
            best_params -- the original method would raise a KeyError here
            if called unmodified.
        """
        if (not self.use_optuna):
            return

        is_continual = (self.config['ContinualLearning']['Replay'].get('use_replay', False)) or (self.config['ContinualLearning']['EWC'].get('use_ewc', False))
        if (is_continual) and (self.current_task.lower() == 'task_a'):
            dataset_type = self.config['Dataset'].get('dataset_type', 'OrganMNIST')
            baseline_metrics_dir = self.base_results_dir / f"{dataset_type}_NoMemory_EWC-False" / "metrics"
            db_path = os.path.join(baseline_metrics_dir, f"{dataset_type}_NoMemory_EWC-False_{self.current_task}.db")
            storage_name = f"sqlite:///{db_path}"
            study = optuna.load_study(study_name=f"{dataset_type}_NoMemory_EWC-False_{self.current_task}", storage=storage_name)
        else:
            db_path = os.path.join(self.metrics_dir, f"{self.exp_id}_{task}.db")
            storage_name = f"sqlite:///{db_path}"
            study = optuna.load_study(study_name=f"{self.exp_id}_{task}", storage=storage_name)

        best_params = study.best_params
        print(f"\n\n==========> Optuna Search Complete <==========")
        print(f"Best Trial Validation Score For Task {task}: {study.best_value:.4f}")
        print("Best Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print("\n\n")

        self.config['Training']['lr'] = best_params['lr']
        self.config['Training']['weight_decay'] = best_params['weight_decay']
        if (self.config['ContinualLearning']['Replay'].get('use_replay', False)) and (self.current_task.lower() != 'task_a'):
            self.config['ContinualLearning']['Replay']['lambda_replay'] = best_params['lambda_replay']
        if (self.config['ContinualLearning']['EWC'].get('use_ewc', False)) and (self.current_task.lower() != 'task_a'):
            self.config['ContinualLearning']['EWC']['lambda_ewc'] = best_params['lambda_ewc']

        if (self.memory_strategy.lower() in ['uncertainty', 'hybrid']) and (task.lower() != "task_a"):
            # ---- ABLATION CHANGE: .get(x, 0.0) instead of best_params[x] ----
            self.config['ContinualLearning']['Replay']['we'] = best_params.get('we', 0.0)
            self.config['ContinualLearning']['Replay']['wH'] = best_params.get('wH', 0.0)
            self.config['ContinualLearning']['Replay']['wa'] = best_params.get('wa', 0.0)
            self.config['ContinualLearning']['Replay']['alea_drop_fraction'] = best_params['alea_drop_fraction']

            if (self.memory_strategy.lower() == 'hybrid'):
                self.config['ContinualLearning']['Replay']['wl'] = best_params['wl']
                self.config['ContinualLearning']['Replay']['pool_multiplier'] = best_params['pool_multiplier']

        if (self.memory_strategy.lower() in ['uncertainty', 'loss', 'hybrid']) and (self.current_task.lower() != "task_a"):
            if (self.config['ContinualLearning']['Replay'].get('optimize_uniform_ratio', True)):
                self.config['ContinualLearning']['Replay']['uniform_ratio'] = best_params['uniform_ratio']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parameters_file', required=True, type=str)
    ap.add_argument('--seed', default=42, type=int)
    ap.add_argument('--replay-memory', action='store_true', default=False)
    args = vars(ap.parse_args())

    parameters_file = args['parameters_file']
    with open(parameters_file, 'r') as file:
        config = yaml.safe_load(file)
    seed = args['seed']
    config['Seed'] = seed
    set_seed(seed)

    replay_memory_only = args['replay_memory']

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

    mem_capacity_samples = int(config['ContinualLearning']['Replay']['capacity'] * data_handler.n_all_train_samples)
    config['ContinualLearning']['Replay']['capacity_in_n_samples'] = mem_capacity_samples

    # ---- ONLY difference from main_experiment.py's main(): uses
    # AblationCLTrainer instead of CLTrainer ----
    trainer = AblationCLTrainer(config=config)

    configs_save_dir = Path(trainer.metrics_dir).parent / "configs"
    configs_save_dir.mkdir(parents=True, exist_ok=True)
    initial_config = deepcopy(trainer.config)
    initial_config_path = configs_save_dir / "initial_config.yaml"
    with open(initial_config_path, 'w') as f:
        yaml.safe_dump(initial_config, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Saved initial configuration checkpoint to: {initial_config_path}")

    tasks = ['Task_A', 'Task_B']
    data_tasks = [task_a_data, task_b_data]
    loader_A, val_loader_A, test_loader_A = trainer.get_data_loaders(task_a_data)
    loader_B, val_loader_B, test_loader_B = trainer.get_data_loaders(task_b_data)
    train_loaders = [loader_A, loader_B]
    val_loaders = [val_loader_A, val_loader_B]
    test_loaders = [test_loader_A, test_loader_B]
    if (trainer.use_optuna):
        for i_task in range(len(tasks)):
            print(f"\n\n==========> Hyper-parameter optimization of task {tasks[i_task]} <==========\n\n")
            current_task = tasks[i_task]
            trainer.current_task = current_task
            if (current_task.lower() != "task_a"):
                trainer.previous_task_data_loader = {
                    'Train': train_loaders[i_task - 1],
                    'Val': val_loaders[i_task - 1],
                    'Test': test_loaders[i_task - 1]
                }
                trainer.previous_task = tasks[i_task - 1]
            else:
                trainer.previous_task_data_loader = None
                trainer.previous_task = None

            trainer.optimize_hyperparameters(data_tasks[i_task], n_trials=config['Optuna'].get('n_trials', 10))
            # NOTE: update_exp_params_optuna() is intentionally NOT called
            # here -- in the original main_experiment.py, it is invoked
            # automatically from inside repeated_holdout() (once per task,
            # right before that task's training begins), not from this
            # loop. Calling it here too would apply it twice.
    else:
        print("\n\n==========> Optuna disabled in config; running directly with configured hyper-parameters <==========\n")

    print("\n\n==========> Starting final full run with optimal configuration <==========\n")
    trainer.repeated_holdout(task_a_data, task_b_data, ext_test_data, save_results=not(replay_memory_only), n_repetitions=config['Training'].get('n_repetitions', 5), replay_memory_only=replay_memory_only)

    final_config_path = configs_save_dir / "final_config.yaml"
    with open(final_config_path, 'w') as f:
        yaml.safe_dump(trainer.config, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Saved final optimized configuration checkpoint to: {final_config_path}")


if __name__ == "__main__":
    main()