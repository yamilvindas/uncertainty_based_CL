"""
    config_generator_ablation.py

    Generates YAML configs for the component-wise ablation study of the
    Proposed Model (Uncertainty Replay + ByClass, WITHOUT EWC).

    Does NOT modify config_generator.py at all -- imports and reuses
    build_base_config() and write_yaml() directly from it.

    Output configs are routed to a separate results folder
    (./results_ablation) via the plain "results_dir" field, so running these
    experiments never touches or overwrites anything in the main ./results
    directory. This requires no code change in main_experiment.py, since
    results_dir is just read straight from the YAML.

    3 components (we, wH, wa) x 3 capacities (1%, 10%, 50%) = 9 configs.
    The "Proposed" (no component ablated) row for these same 3 capacities
    is NOT regenerated here -- it already exists in the main experiment
    matrix (Uncertainty Replay + ByClass, no EWC) and should be reused
    directly from ./results when building the final ablation table.

    Usage:
        python src/utils/config_generator_ablation.py \\
            --HITS-data-path-A path/to/data_A.hdf5 \\
            --HITS-data-path-B path/to/data_B.hdf5
"""
import argparse
import os

from config_generator import build_base_config, write_yaml

ABLATION_CAPACITIES = {
    "Mem-1": 0.01,
    "Mem-10": 0.10,
    "Mem-50": 0.50,
}
ABLATE_COMPONENTS = ["we", "wH", "wa"]
RESULTS_DIR = "./results_ablation"


def generate_ablation_configs(hits_data_path_A, hits_data_path_B, hits_data_path_EXT=None):
    base_dir = "configs/HITS/Ablation"

    for mem_folder, capacity in ABLATION_CAPACITIES.items():
        for component in ABLATE_COMPONENTS:
            cfg = build_base_config(
                "HITS",
                hits_data_paths=[hits_data_path_A, hits_data_path_B, hits_data_path_EXT],
            )

            # Redirect output to a SEPARATE results folder -- does not touch
            # or overwrite anything under the main results/ directory.
            cfg["results_dir"] = RESULTS_DIR

            # exp_id: capacity / strategy / EWC / by_class are all
            # auto-appended by CLTrainer.__init__ in main_experiment.py, so
            # only the ablation-specific prefix is needed here.
            cfg["exp_id"] = f"HITS_Ablation_x{component}"

            # ---- Proposed Model = Uncertainty Replay + ByClass, NO EWC ----
            cfg["ContinualLearning"]["Replay"]["use_replay"] = True
            cfg["ContinualLearning"]["Replay"]["memory_strategy"] = "uncertainty"
            cfg["ContinualLearning"]["Replay"]["capacity"] = capacity
            cfg["ContinualLearning"]["Replay"]["lambda_replay"] = 1.0
            cfg["ContinualLearning"]["Replay"]["by_class"] = True
            cfg["ContinualLearning"]["EWC"]["use_ewc"] = False
            cfg["ContinualLearning"]["EWC"]["lambda_ewc"] = 0.0

            # Base uncertainty-score hyperparameters (same defaults used in
            # config_generator.py before Optuna overrides them at runtime)
            cfg["ContinualLearning"]["Replay"]["we"] = 1.0
            cfg["ContinualLearning"]["Replay"]["wH"] = 1.0
            cfg["ContinualLearning"]["Replay"]["wa"] = 1.0
            cfg["ContinualLearning"]["Replay"]["alea_drop_fraction"] = 0.15
            cfg["ContinualLearning"]["Replay"]["mc_passes"] = 20
            cfg["ContinualLearning"]["Replay"]["optimize_uniform_ratio"] = False
            cfg["ContinualLearning"]["Replay"]["uniform_ratio"] = 0.5

            # Matches existing convention: 75 trials for uncertainty-based
            # strategies (larger Optuna search space)
            cfg["Optuna"]["n_trials"] = 75

            # NEW field, ADDITIVE ONLY -- ignored by the original,
            # unmodified main_experiment.py; read only by the new
            # main_experiment_ablation.py
            cfg["Ablation"] = {"ablate_component": component}

            out_path = os.path.join(base_dir, mem_folder, f"Ablation-x{component}.yaml")
            write_yaml(out_path, cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--HITS-data-path-A", required=True)
    ap.add_argument("--HITS-data-path-B", required=True)
    ap.add_argument("--HITS-data-path-EXT", default=None)
    args = vars(ap.parse_args())

    generate_ablation_configs(
        hits_data_path_A=args["HITS_data_path_A"],
        hits_data_path_B=args["HITS_data_path_B"],
        hits_data_path_EXT=args["HITS_data_path_EXT"],
    )
    print("\n=======> 9 ablation config files generated in "
          "configs/HITS/Ablation/{Mem-1,Mem-10,Mem-50}/ <=======\n")