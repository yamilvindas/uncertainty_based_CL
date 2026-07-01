import glob
import os
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)

# -------------------------------------------------------------------------
# 1. METRIC EXTRACTION HELPER FUNCTIONS
# -------------------------------------------------------------------------
METRIC_TABLE_NAME = {
    'Acc': 'Acc',
    'BalAcc': 'Bal. Acc',
    'MCC': 'MCC',
}

def calculate_metrics(targets, preds):
    return {
        'Acc': accuracy_score(targets, preds),
        'BalAcc': balanced_accuracy_score(targets, preds),
        'MCC': matthews_corrcoef(targets, preds),
        'F1_Macro': f1_score(targets, preds, average='macro', zero_division=0)
    }

def parse_h5_file(h5_path):
    """
        Extracts mean and std of key metrics across all repetitions in an H5 file.
    """
    results = {}
    try:
        with h5py.File(h5_path, 'r') as f:
            rep_keys = [k for k in f.keys() if k.startswith('Rep_')]
            if (not rep_keys):
                print(f"[ERROR] No 'Rep_X' groups found in {h5_path}. Was the training completed?")
                return None
            
            # Temporary storage for metrics across reps
            metrics_storage = {
                                'Final_Acc_Task_A': [],
                                'Final_Acc_Task_B': [],
                                'Forgetting_Acc_Task_A': [],
                                'BWT_Acc_Task_A': [],
                                'Final_BalAcc_Task_A': [],
                                'Final_BalAcc_Task_B': [],
                                'Forgetting_BalAcc_Task_A': [],
                                'BWT_BalAcc_Task_A': [],
                                'Final_MCC_Task_A': [],
                                'Final_MCC_Task_B': [],
                                'Forgetting_MCC_Task_A': [],
                                'BWT_MCC_Task_A': []
                            }
            
            for rep_key in rep_keys:
                rep_group = f[rep_key]
                
                # Check if phase groups exist
                if ('Phase_Post_Task_A' not in rep_group) or ('Phase_Post_Task_B' not in rep_group):
                    continue
                
                post_A_group = rep_group['Phase_Post_Task_A']
                post_B_group = rep_group['Phase_Post_Task_B']
                
                # Ensure test sets exist
                if ('Test_Task_A' not in post_A_group) or ('Test_Task_A' not in post_B_group) or ('Test_Task_B' not in post_B_group):
                    continue
                    
                # Task A after Task A
                targets_A = post_A_group['Test_Task_A']['targets'][:]
                preds_A = post_A_group['Test_Task_A']['preds'][:]
                metrics_A = calculate_metrics(targets_A, preds_A)
                
                # Task A after Task B
                targets_B_A = post_B_group['Test_Task_A']['targets'][:]
                preds_B_A = post_B_group['Test_Task_A']['preds'][:]
                metrics_B_A = calculate_metrics(targets_B_A, preds_B_A)

                # Task B after Task B
                targets_B = post_B_group['Test_Task_B']['targets'][:]
                preds_B = post_B_group['Test_Task_B']['preds'][:]
                metrics_B = calculate_metrics(targets_B, preds_B)
                
                # Accuracy metrics
                for metrics_name in ['Acc', 'BalAcc', 'MCC']:
                    metrics_storage[f'Final_{metrics_name}_Task_A'].append(metrics_B_A[metrics_name])
                    metrics_storage[f'Final_{metrics_name}_Task_B'].append(metrics_B[metrics_name])
                    metrics_storage[f'Forgetting_{metrics_name}_Task_A'].append(metrics_A[metrics_name] - metrics_B_A[metrics_name])
                    metrics_storage[f'BWT_{metrics_name}_Task_A'].append(metrics_B_A[metrics_name] - metrics_A[metrics_name])

            # Calculate Means and Stds
            for k, v in metrics_storage.items():
                if len(v) > 0:
                    results[f"{k}_Mean"] = np.mean(v)
                    results[f"{k}_Std"] = np.std(v)
                else:
                    print(f"[ERROR] No valid data for {h5_path}: {k} has no values.")
                    return None
            return results
    except Exception as e:
        print(f"[ERROR] Error parsing {h5_path}: {e}")
        return None

def extract_metadata(folder_name):
    """
        Parses the experiment ID folder name to extract method details.
    """
    dataset = "Unknown"
    if ("OrganMNIST" in folder_name):
        dataset = "OrganMNIST"
    elif ("Camelyon17" in folder_name):
        dataset = "Camelyon17"
    elif ("HITS" in folder_name):
        dataset = "HITS"
    
    is_ewc = "EWC-True" in folder_name
    
    if ("MemStrategy" in folder_name):
        # Strategy name
        strategy_match = re.search(r"MemStrategy-([^_]+)", folder_name)
        capacity_match = re.search(r"MemCapacity-([^_]+)", folder_name)
        
        strategy = strategy_match.group(1).capitalize() if strategy_match else "Unknown"
        capacity = float(capacity_match.group(1)) if capacity_match else 0.0

        approach_name = f"{strategy} Replay"
        
        # By class approach?
        if ('ByClass-True' in folder_name):
            approach_name += " + ByClass"
        
        # EWC combined?
        
        if (is_ewc):
            approach_name += " + EWC"
        
        group = f"Mem-{int(capacity * 100)}%"
    else:
        capacity = 0.0
        if is_ewc:
            approach_name = "EWC Only"
            group = "Baselines"
        else:
            approach_name = "Baseline (No Memory)"
            group = "Baselines"

            
    return dataset, approach_name, group, capacity, is_ewc

# -------------------------------------------------------------------------
# 2. MAIN REPORT GENERATION
# -------------------------------------------------------------------------
def generate_reports(metrics_to_show=['Acc', 'BalAcc', 'MCC'], results_dir="./results", output_dir="./report"):
    """
        Generated with Gemini 3.1 Pro
    """
    os.makedirs(output_dir, exist_ok=True)
    
    h5_files = glob.glob(os.path.join(results_dir, "*", "metrics", "predictions*.h5"))
    if not h5_files:
        print(f"No predictions.h5 files found in {results_dir}")
        return

    data = []

    # Process all files
    for h5_path in h5_files:
        exp_folder = os.path.basename(os.path.dirname(os.path.dirname(h5_path)))
        dataset, approach, group, capacity, is_ewc = extract_metadata(exp_folder)
        
        metrics = parse_h5_file(h5_path)
        if metrics is not None:
            row = {
                'Dataset': dataset,
                'Group': group,
                'Approach': approach,
                'Capacity': capacity,
                'Is_EWC': is_ewc,
                **metrics
            }
            data.append(row)
            
    df = pd.DataFrame(data)
    if df.empty:
        print("No valid data could be parsed.")
        return
    df.to_csv(os.path.join(output_dir, "aggregated_results.csv"), index=False)

    # Helper: Format mean/std for LaTeX
    def format_mean_std(row, metric):
        mean = row[f'{metric}_Mean'] * 100
        std = row[f'{metric}_Std'] * 100
        return f"{mean:.2f} \\pm {std:.2f}"

    # -------------------------------------------------------------------------
    # 3. LATEX TABLE GENERATION
    # -------------------------------------------------------------------------
    for dataset in df['Dataset'].unique():
        ds_df = df[df['Dataset'] == dataset].copy()
        
        # Sort values: Baselines first, then by capacity, then by strategy
        # Create a categorical order for the Groups
        unique_caps = sorted(ds_df[ds_df['Capacity'] > 0]['Capacity'].unique())
        group_order = ["Baselines"] + [f"Mem-{int(c*100)}%" for c in unique_caps]
        ds_df['Group'] = pd.Categorical(ds_df['Group'], categories=group_order, ordered=True)
        ds_df = ds_df.sort_values(['Group', 'Capacity', 'Approach'])
        
        column_names = []
        for m in metrics_to_show:
            column_names.extend([
                f"{METRIC_TABLE_NAME[m]} Task A (\\%)",
                f"{METRIC_TABLE_NAME[m]} Task B (\\%)",
                f"Forgetting {METRIC_TABLE_NAME[m]} (\\%)"
            ])
        column_line = "} & \\textbf{".join(column_names)
        tabular_align = "ll" + "ccc" * len(metrics_to_show)
        
        latex_lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Continual Learning Results on " + dataset + "}",
            "\\label{tab:results_" + dataset.lower() + "}",
            f"\\begin{{tabular}}{{{tabular_align}}}",
            "\\toprule",
            "\\textbf{Memory} & \\textbf{Approach} & \\textbf{" + column_line + "} \\\\",
            "\\midrule"
        ]
        
        current_group = None
        for _, row in ds_df.iterrows():
            if (row['Group'] != current_group):
                if current_group is not None:
                    latex_lines.append("\\midrule")
                current_group = row['Group']
                if (current_group == 'Baselines'):
                    n_rows = 2
                else:
                    n_rows = 4
                if ('%' in current_group):
                    tmp_current_group = current_group.replace('%', '\\%')
                else:
                    tmp_current_group = current_group
                latex_lines.append(f"\\multirow{{{n_rows}}}{{*}}{{{tmp_current_group}}} ")
            
            metric_line = f"& {row['Approach']} "
            for m in metrics_to_show:
                metric_a = format_mean_std(row, f'Final_{m}_Task_A')
                metric_b = format_mean_std(row, f'Final_{m}_Task_B')
                forgetting = format_mean_std(row, f'Forgetting_{m}_Task_A')
                metric_line += f"& ${metric_a}$ & ${metric_b}$ & ${forgetting}$ "
            metric_line += "\\\\"
            
            latex_lines.append(metric_line)
            
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}"
        ])
        
        tex_path = os.path.join(output_dir, f"{dataset}_results_table.tex")
        with open(tex_path, 'w') as f:
            f.write("\n".join(latex_lines))
        print(f"LaTeX table generated: {tex_path}")

    # -------------------------------------------------------------------------
    # 4. FIGURE GENERATION (TREND PLOTS)
    # -------------------------------------------------------------------------
    sns.set_theme(style="whitegrid", palette="muted")
    
    for dataset in df['Dataset'].unique():
        ds_df = df[df['Dataset'] == dataset].copy()
        
        # Split into baselines and replay methods
        baselines = ds_df[ds_df['Capacity'] == 0]
        replay_df = ds_df[ds_df['Capacity'] > 0].copy()
        
        if replay_df.empty:
            continue
            
        # Clean up approach names for the legend (Remove "+ EWC" to use it as a line style)
        replay_df['Strategy'] = replay_df['Approach'].apply(lambda x: x.replace(" + EWC", ""))
        
        metrics_to_plot = {
            'Final_Acc_Task_A_Mean': ('Final Accuracy - Task A', 'higher'),
            'Final_Acc_Task_B_Mean': ('Final Accuracy - Task B', 'higher'),
            'Forgetting_Acc_Task_A_Mean': ('Catastrophic Forgetting - Task A', 'lower'),
            # Balanced Accuracy trend plots
            'Final_BalAcc_Task_A_Mean': ('Final Balanced Accuracy - Task A', 'higher'),
            'Final_BalAcc_Task_B_Mean': ('Final Balanced Accuracy - Task B', 'higher'),
            'Forgetting_BalAcc_Task_A_Mean': ('Catastrophic Forgetting Bal.Acc - Task A', 'lower'),
            # MCC trend plots
            'Final_MCC_Task_A_Mean': ('Final MCC - Task A', 'higher'),
            'Final_MCC_Task_B_Mean': ('Final MCC - Task B', 'higher'),
            'Forgetting_MCC_Task_A_Mean': ('Catastrophic Forgetting MCC - Task A', 'lower')
        }
        
        for metric_col, (metric_label, direction) in metrics_to_plot.items():
            plt.figure(figsize=(10, 6))

            # 1. Plot the Replay Methods
            sns.lineplot(
                data=replay_df, 
                x='Capacity', 
                y=metric_col, 
                hue='Strategy',
                style='Is_EWC',
                markers=True, 
                dashes=True,
                linewidth=2,
                markersize=8
            )
            
            
            # 2. Add Horizontal Lines for Baselines
            if not baselines.empty:
                for _, row in baselines.iterrows():
                    color = 'red' if 'No Memory' in row['Approach'] else 'black'
                    ls = '--' if 'No Memory' in row['Approach'] else ':'
                    plt.axhline(
                        y=row[metric_col], 
                        color=color, 
                        linestyle=ls, 
                        label=f"{row['Approach']} Baseline",
                        alpha=0.7
                    )
            
            plt.title(f"{dataset}: {metric_label} vs Memory Capacity", fontsize=14, pad=15)
            plt.xlabel("Memory Capacity Fraction (0.0 to 1.0)", fontsize=12)
            plt.ylabel(metric_label, fontsize=12)
            
            # Adjust X-axis ticks to match tested capacities
            capacities = sorted(replay_df['Capacity'].unique())
            plt.xticks(capacities, [f"{int(c*100)}%" for c in capacities])
            
            # Clean legend
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Approaches")
            plt.tight_layout()
            
            # Save plot
            metric_clean_name = metric_col.replace('_Mean', '')
            plot_path = os.path.join(output_dir, f"{dataset}_Trend_{metric_clean_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Trend plot generated: {plot_path}")

    # -------------------------------------------------------------------------
    # 5. BAR PLOTS (Specific Memory Capacity Compare)
    # -------------------------------------------------------------------------
    # Generate a bar plot specifically for the 10% memory capacity mark
    target_capacity = 0.10
    for dataset in df['Dataset'].unique():
        ds_df = df[(df['Dataset'] == dataset) & ((df['Capacity'] == target_capacity) | (df['Capacity'] == 0))]
        if ds_df.empty: continue
        
        plt.figure(figsize=(12, 6))
        
        # Sort by approach to make it look clean
        ds_df = ds_df.sort_values('Approach')
        
        sns.barplot(
            data=ds_df,
            x='Approach',
            y='Final_Acc_Task_A_Mean',
            hue='Approach',
            legend=False,
            palette="viridis"
        )
        
        # Add error bars manually if desired, or let seaborn handle raw data if we exploded the dataframe
        # Here we use the precomputed means, so we just add error bars using matplotlib's errorbar
        x_coords = np.arange(len(ds_df))
        plt.errorbar(
            x=x_coords,
            y=ds_df['Final_Acc_Task_A_Mean'],
            yerr=ds_df['Final_Acc_Task_A_Std'],
            fmt='none',
            c='black',
            capsize=5
        )
        
        plt.title(f"{dataset}: Task A Final Accuracy Comparison (at {int(target_capacity*100)}% Memory)", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Accuracy", fontsize=12)
        plt.tight_layout()
        
        bar_path = os.path.join(output_dir, f"{dataset}_BarPlot_Mem10_AccA.png")
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bar plot generated: {bar_path}")

        # Additional bar plot for Balanced Accuracy at 10%
        plt.figure(figsize=(12, 6))
        ds_df = ds_df.sort_values('Approach')

        sns.barplot(
            data=ds_df,
            x='Approach',
            y='Final_BalAcc_Task_A_Mean',
            hue='Approach',
            legend=False,
            palette="viridis"
        )

        x_coords = np.arange(len(ds_df))
        plt.errorbar(
            x=x_coords,
            y=ds_df['Final_BalAcc_Task_A_Mean'],
            yerr=ds_df['Final_BalAcc_Task_A_Std'],
            fmt='none',
            c='black',
            capsize=5
        )

        plt.title(f"{dataset}: Task A Final Balanced Accuracy Comparison (at {int(target_capacity*100)}% Memory)", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Balanced Accuracy", fontsize=12)
        plt.tight_layout()

        bal_bar_path = os.path.join(output_dir, f"{dataset}_BarPlot_Mem10_BalAccA.png")
        plt.savefig(bal_bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bar plot generated: {bal_bar_path}")

        # MCC bar plot at 10%
        plt.figure(figsize=(12, 6))
        ds_df = ds_df.sort_values('Approach')

        sns.barplot(
            data=ds_df,
            x='Approach',
            y='Final_MCC_Task_A_Mean',
            hue='Approach',
            legend=False,
            palette="viridis"
        )

        x_coords = np.arange(len(ds_df))
        plt.errorbar(
            x=x_coords,
            y=ds_df['Final_MCC_Task_A_Mean'],
            yerr=ds_df['Final_MCC_Task_A_Std'],
            fmt='none',
            c='black',
            capsize=5
        )

        plt.title(f"{dataset}: Task A Final MCC Comparison (at {int(target_capacity*100)}% Memory)", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("MCC", fontsize=12)
        plt.tight_layout()

        mcc_bar_path = os.path.join(output_dir, f"{dataset}_BarPlot_Mem10_MCC_A.png")
        plt.savefig(mcc_bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bar plot generated: {mcc_bar_path}")

    print("\nReport generation completed successfully! All files are in the './report' directory.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reports from H5 prediction files.")
    parser.add_argument('--m_to_show', help='Metrics to include in the LaTeX table', nargs='+', default=['Acc', 'BalAcc', 'MCC'])
    args = parser.parse_args()
    
    metrics_to_show = args.m_to_show
    for m in metrics_to_show:
        if m not in METRIC_TABLE_NAME:
            print(f"[ERROR] Invalid metric '{m}' specified. Valid options are: {list(METRIC_TABLE_NAME.keys())}")
            exit(1)

    generate_reports(metrics_to_show=metrics_to_show)