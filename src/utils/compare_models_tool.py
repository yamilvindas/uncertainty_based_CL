import h5py
import numpy as np
import argparse
from pathlib import Path
from scipy import stats
from sklearn.metrics import accuracy_score
import warnings

# Suppress sklearn undefined metric warnings for 0 division
warnings.filterwarnings('ignore', category=UserWarning)

# Internal imports
from src.utils.report_generator import METRIC_TABLE_NAME, calculate_metrics

def get_metrics_from_h5(h5_path):
    """
        Extracts the core Continual Learning metrics
        for every repetition inside the HDF5 file.
    """
    path = Path(h5_path)
    if (not path.exists()):
        raise FileNotFoundError(f"File not found: {h5_path}")

    metrics = {
        'Final_Acc_A': [],
        'Final_Acc_B': [],
        'Forgetting_A': []
    }
    metrics = {
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
    
    with h5py.File(path, 'r') as f:
        exp_name = f.attrs.get('experiment_name', path.parent.parent.name)
        rep_keys = [k for k in f.keys() if k.startswith('Rep_')]
        
        for rep_key in rep_keys:
            rep_group = f[rep_key]

            # Check if necessary phases exist
            if ('Phase_Post_Task_A' not in rep_group) or ('Phase_Post_Task_B' not in rep_group):
                continue
                
            post_A_group = rep_group['Phase_Post_Task_A']
            post_B_group = rep_group['Phase_Post_Task_B']
            
            if ('Test_Task_A' not in post_A_group) or ('Test_Task_A' not in post_B_group) or ('Test_Task_B' not in post_B_group):
                continue
                
            # Metric A after learning A
            t_A_A = post_A_group['Test_Task_A']['targets'][:]
            p_A_A = post_A_group['Test_Task_A']['preds'][:]
            metrics_A_A = calculate_metrics(t_A_A, p_A_A)
            
            # Metric A after learning B (Final Metric A)
            t_B_A = post_B_group['Test_Task_A']['targets'][:]
            p_B_A = post_B_group['Test_Task_A']['preds'][:]
            metrics_B_A = calculate_metrics(t_B_A, p_B_A)
            
            # Metric B after learning B (Final Metric B)
            t_B_B = post_B_group['Test_Task_B']['targets'][:]
            p_B_B = post_B_group['Test_Task_B']['preds'][:]
            metrics_B_B = calculate_metrics(t_B_B, p_B_B)

            # Extract the relevant metrics
            for metric_name in METRIC_TABLE_NAME.keys():
                metrics[f'Final_{metric_name}_Task_A'].append(metrics_B_A[metric_name])
                metrics[f'Final_{metric_name}_Task_B'].append(metrics_B_B[metric_name])
                metrics[f'Forgetting_{metric_name}_Task_A'].append(metrics_A_A[metric_name] - metrics_B_A[metric_name])
                metrics[f'BWT_{metric_name}_Task_A'].append(metrics_B_A[metric_name] - metrics_A_A[metric_name])            

    return exp_name, metrics

def evaluate_significance(vals_1, vals_2, metric_name, alpha, higher_is_better=True):
    """
        Performs the Mann-Whitney U test and formats the output.
        Uses the Bonferroni-adjusted alpha threshold.
    """
    # Get means and standard deviations
    mean1, std1 = np.mean(vals_1), np.std(vals_1)
    mean2, std2 = np.mean(vals_2), np.std(vals_2)
    
    # Run two-sided Mann-Whitney U test
    stat, p_value = stats.mannwhitneyu(vals_1, vals_2, alternative='two-sided')
    
    # Test for significance
    is_significant = p_value < alpha
    
    # Determine the "Winner" if significant
    winner = "Tie / Inconclusive"
    if is_significant:
        if (mean1 > mean2 and higher_is_better) or (mean1 < mean2 and not higher_is_better):
            winner = "Model 1"
        else:
            winner = "Model 2"
            
    print(f"\n--- {metric_name.upper().replace('_', ' ')} ---")
    print(f" \t===> Model 1 : {mean1:6.2f} ± {std1:.2f}")
    print(f" \t===> Model 2 : {mean2:6.2f} ± {std2:.2f}")
    print(f" \t===> P-Value : {p_value:.5f} (Threshold: < {alpha:.5f})")
    
    if is_significant:
        print(f"\t===> Result: SIGNIFICANT DIFFERENCE => {winner} is better.")
    else:
        print(f"\t===> Result: Not statistically significant.")


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
    ap.add_argument('--model_1_h5', type=str, help="Path to the predictions_0.h5 file for Model 1")
    ap.add_argument('--model_2_h5', type=str, help="Path to the predictions_0.h5 file for Model 2")
    ap.add_argument("--bonferroni_n", type=int, default=3, help="Number of comparisons for Bonferroni correction")
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    model_1_h5 = args['model_1_h5']
    model_2_h5 = args['model_2_h5']
    bonferroni_n = args['bonferroni_n']

    #====================================================================================================#
    #====================================== Bonferroni Correction ======================================#
    #====================================================================================================#
    # Bonferroni adjustment
    base_alpha = 0.05
    adjusted_alpha = base_alpha / bonferroni_n

    #====================================================================================================#
    #====================================== Statistical Comparison ======================================#
    #====================================================================================================#
    print("=" * 70)
    print("===> Continual learning statistical comparison <===")
    print("=" * 70)
    print(f"===> Bonferroni Correction : Applied (n={bonferroni_n})")
    print(f"===> Adjusted Alpha        : {adjusted_alpha:.5f}")

    # Metric to use 
    #METRIC_TO_USE = "Acc"
    METRIC_TO_USE = "BalAcc"
    #METRIC_TO_USE = "MCC"

    # Get metrics
    try:
        name1, metrics1 = get_metrics_from_h5(model_1_h5)
        name2, metrics2 = get_metrics_from_h5(model_2_h5)
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    print(f"===> Model 1: {name1} (n={len(metrics1[f'Final_{METRIC_TO_USE}_Task_A'])})")
    print(f"===> Model 2: {name2} (n={len(metrics2[f'Final_{METRIC_TO_USE}_Task_A'])})")
    print("-" * 70)

    # Verify that both models have at least one repetition
    if (len(metrics1[f'Final_{METRIC_TO_USE}_Task_A']) == 0) or (len(metrics2[f'Final_{METRIC_TO_USE}_Task_A']) == 0):
        print("[ERROR] One or both models have 0 completed repetitions parsed.")
        return

    # Compare metrics for Task A
    evaluate_significance(
        metrics1[f'Final_{METRIC_TO_USE}_Task_A'], 
        metrics2[f'Final_{METRIC_TO_USE}_Task_A'], 
        f"Final_{METRIC_TO_USE}_Task_A",  
        adjusted_alpha,
        higher_is_better=True
    )
    # Compare metrics for Task B
    evaluate_significance(
        metrics1[f'Final_{METRIC_TO_USE}_Task_B'], 
        metrics2[f'Final_{METRIC_TO_USE}_Task_B'], 
        f"Final_{METRIC_TO_USE}_Task_B",  
        adjusted_alpha,
        higher_is_better=True
    )
    # Compare forgetting metrics
    evaluate_significance(
        metrics1[f'Forgetting_{METRIC_TO_USE}_Task_A'], 
        metrics2[f'Forgetting_{METRIC_TO_USE}_Task_A'], 
        f"Forgetting_{METRIC_TO_USE}_Task_A",  
        adjusted_alpha,
        higher_is_better=False
    )
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()