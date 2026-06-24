#!/bin/bash

# Exit immediately if a command exits with a non-zero status,
# but allow graceful handling of interruptions (SIGINT)
set -e

CONFIGS_DIR="configs"
PROGRESS_LOG="completed_runs.log"
MAIN_SCRIPT="src/experiments/main_experiment.py"  # Your main training entrypoint script

# Activate environment
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:./uncertainty_based_CL/"

# Ensure the progress log exists
touch "$PROGRESS_LOG"

# Clean shutdown on Ctrl+C (SIGINT)
cleanup() {
    echo -e "\n\n[INFO] Script execution interrupted by user. Exiting gracefully..."
    exit 130
}
trap cleanup SIGINT

echo "=========================================================="
echo "    Continual Learning Ordered Runner (With Resume)       "
echo "=========================================================="
echo "Enforcing execution order:"
echo " 1. Datasets: OrganMNIST -> Camelyon17 -> HITS"
echo " 2. Approaches: Baseline -> Mem-1 -> ... -> Mem-100"
echo " 3. Memory Strategies: uniform -> loss -> dissimilarity -> uncertainty"
echo " 4. Modifiers: No EWC (NoMemory/Standard Replay) -> With EWC"
echo "----------------------------------------------------------"

# Define the precise datasets and folder sequence to build the queue
DATASETS=("OrganMNIST" "Camelyon17" "HITS")
SEQUENCE=("Baseline" "Mem-1" "Mem-5" "Mem-10" "Mem-50" "Mem-100")

# 1. Build the ordered queue of configuration files
QUEUE=()
for ds in "${DATASETS[@]}"; do
    for seq_folder in "${SEQUENCE[@]}"; do
        target_dir="$CONFIGS_DIR/$ds/$seq_folder"
        
        if [ -d "$target_dir" ]; then
            # Unified Python sorter for ALL directories
            ordered_files=$(python3 -c '
import os, sys, yaml
target_dir = sys.argv[1]
# Desired order index for strategies
strategies_order = {"uniform": 0, "loss": 1, "dissimilarity": 2, "uncertainty": 3}

files = [f for f in os.listdir(target_dir) if f.endswith(".yaml") or f.endswith(".yml")]
file_info = []

for f in files:
    path = os.path.join(target_dir, f)
    try:
        with open(path, "r") as file:
            cfg = yaml.safe_load(file)
            
            # Extract Continual Learning values safely
            cl_cfg = cfg.get("ContinualLearning", {})
            replay_cfg = cl_cfg.get("Replay", {})
            ewc_cfg = cl_cfg.get("EWC", {})
            
            use_replay = replay_cfg.get("use_replay", False)
            use_ewc = ewc_cfg.get("use_ewc", False)
            
            # Strategy ordering: Baselines map to -1 so they are handled cleanly
            if use_replay:
                strategy = replay_cfg.get("memory_strategy", "uniform").lower()
                strat_idx = strategies_order.get(strategy, 99)
            else:
                strat_idx = -1
                
            # Tuple: (Strategy Rank, EWC Rank (False/0 before True/1), File Path)
            file_info.append((strat_idx, int(use_ewc), path))
    except Exception as e:
        # Fallback for unparseable files
        file_info.append((99, 0, path))

# Sort by strategy first, then EWC status, then alphabetical path
file_info.sort(key=lambda x: (x[0], x[1], x[2]))

# Output ordered paths to bash
for info in file_info:
    print(info[2])
' "$target_dir")
            
            # Append the Python-ordered files to the bash array
            while IFS= read -r file; do
                if [ -n "$file" ]; then
                    QUEUE+=("$file")
                fi
            done <<< "$ordered_files"
        fi
    done
done

total_configs=${#QUEUE[@]}
if [ "$total_configs" -eq 0 ]; then
    echo "[ERROR] No configuration files found in sequence. Please verify your folder generation."
    exit 1
fi

echo "Successfully queued $total_configs experiment configurations."
echo "----------------------------------------------------------"

run_count=0
skip_count=0
idx=0

# 2. Iterate through our custom ordered queue
for config_file in "${QUEUE[@]}"; do
    idx=$((idx + 1))
    
    # Extract metadata safely using a robust Python execution with single quotes
    metadata=$(python3 -c '
import yaml, sys
try:
    with open(sys.argv[1], "r") as f:
        cfg = yaml.safe_load(f)
        
        res_dir = cfg.get("results_dir", "./results")
        exp_id = cfg.get("exp_id", "unknown")
        
        # Build strict dynamic EXP_ID matching your main logic
        cl_cfg = cfg.get("ContinualLearning", {})
        
        # Replay memory tags
        if cl_cfg.get("Replay", {}).get("use_replay", False):
            mem_strategy = cl_cfg["Replay"].get("memory_strategy", "Uniform")
            mem_capacity = cl_cfg["Replay"].get("capacity", 0.1)
            exp_id += f"_MemStrategy-{mem_strategy}_MemCapacity-{mem_capacity}"
            
        # EWC tags
        if cl_cfg.get("EWC", {}).get("use_ewc", False):
            exp_id += "_EWC-True"
        else:
            exp_id += "_EWC-False"
            
        print(f"{res_dir}|{exp_id}")
except Exception as e:
    print("ERROR")
' "$config_file" 2>/dev/null)

    if [ "$metadata" = "ERROR" ] || [ -z "$metadata" ]; then
        echo "[WARNING] Could not parse configuration file: $config_file. Skipping."
        continue
    fi

    # Split the metadata
    RESULTS_DIR=$(echo "$metadata" | cut -d'|' -f1)
    EXP_ID=$(echo "$metadata" | cut -d'|' -f2)
    
    # Path where HDF5 predictions are located
    TARGET_H5="$RESULTS_DIR/$EXP_ID/metrics/predictions_0.h5"

    # Progress/Resume verification conditions:
    if grep -qF "$config_file" "$PROGRESS_LOG" && [ -s "$TARGET_H5" ]; then
        echo "[$idx/$total_configs] [SKIP] Completed -> $EXP_ID"
        skip_count=$((skip_count + 1))
        continue
    fi

    # Fallback disk-check in case the log missed a run but the output exists
    if [ -s "$TARGET_H5" ]; then
        echo "[$idx/$total_configs] [SKIP] File exists on disk -> $EXP_ID"
        echo "$config_file" >> "$PROGRESS_LOG"
        skip_count=$((skip_count + 1))
        continue
    fi

    echo ""
    echo "=========================================================="
    echo " PROGRESS: $idx / $total_configs"
    echo " RUNNING : $EXP_ID"
    echo " CONFIG  : $config_file"
    echo " TARGET  : $TARGET_H5"
    echo "=========================================================="

    # Run the model training pipeline
    set +e
    printf "python %s --parameters_file %s\n" "$MAIN_SCRIPT" "$config_file"
    python3 "$MAIN_SCRIPT" --parameters_file "$config_file"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[SUCCESS] Finished: $EXP_ID"
        echo "$config_file" >> "$PROGRESS_LOG"
        run_count=$((run_count + 1))
    else
        echo "[ERROR] Experiment failed with exit code $EXIT_CODE on config: $config_file"
        echo "The queue was paused. Once debugged, run this script again."
        echo "It will automatically pick up from this exact step."
        exit $EXIT_CODE
    fi
done

echo ""
echo "=========================================================="
echo "                 EXPERIMENT RUN COMPLETE                  "
echo "=========================================================="
echo "Total Configs Evaluated : $total_configs"
echo "Newly Executed Runs     : $run_count"
echo "Skipped (Already Done)  : $skip_count"
echo "=========================================================="