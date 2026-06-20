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

# Clean clean shutdown on Ctrl+C (SIGINT)
cleanup() {
    echo -e "\n\n[INFO] Script execution interrupted by user. Exiting gracefully..."
    exit 130
}
trap cleanup SIGINT

echo "=========================================================="
echo "    Continual Learning Ordered Runner (With Resume)       "
echo "=========================================================="
echo "Enforcing execution order:"
echo " 1. OrganMNIST (Baselines -> Mem-1 -> ... -> Mem-100)"
echo " 2. Camelyon17  (Baselines -> Mem-1 -> ... -> Mem-100)"
echo "----------------------------------------------------------"

# Define the precise datasets and folder sequence to build the queue
DATASETS=("OrganMNIST" "Camelyon17")
SEQUENCE=("Baseline" "Mem-1" "Mem-5" "Mem-10" "Mem-50" "Mem-100")

# 1. Build the ordered queue of configuration files
QUEUE=()
for ds in "${DATASETS[@]}"; do
    for seq_folder in "${SEQUENCE[@]}"; do
        target_dir="$CONFIGS_DIR/$ds/$seq_folder"
        
        if [ -d "$target_dir" ]; then
            # Find and sort files in this directory to maintain a stable alphabetical run order
            # (e.g., running Replay.yaml before ReplayEWC.yaml)
            while IFS= read -r file; do
                if [ -n "$file" ]; then
                    QUEUE+=("$file")
                fi
            done < <(find "$target_dir" -maxdepth 1 -type f \( -name "*.yaml" -o -name "*.yml" \) | sort)
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
    
    # Extract metadata safely using a fast inline Python execution
    metadata=$(python -c "
import yaml
try:
    with open('$config_file', 'r') as f:
        cfg = yaml.safe_load(f)
        # Results directory
        res_dir = cfg['results_dir']
        # Exp ID as defined in main_experiments.py
        exp_id = cfg['exp_id']
        # Replay memory
        if (cfg['ContinualLearning']['Replay'].get('use_replay', False)):
            mem_strategy = cfg['ContinualLearning']['Replay'].get('memory_strategy', 'Uniform')
            mem_capacity = cfg['ContinualLearning']['Replay']['capacity']
            exp_id += f'_MemStrategy-{mem_strategy}_MemCapacity-{mem_capacity}'
        # EWC
        if (cfg['ContinualLearning']['EWC'].get('use_ewc', False)):
            exp_id += '_EWC-True'
        else:
            exp_id += '_EWC-False'
        print(f'{res_dir}|{exp_id}')
except Exception as e:
    print('ERROR')
" 2>/dev/null)


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
    # Check if recorded in the log, and verify the physical HDF5 file exists and is not empty.
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
    printf "python $MAIN_SCRIPT --parameters_file $config_file"
    python "$MAIN_SCRIPT" --parameters_file "$config_file"
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