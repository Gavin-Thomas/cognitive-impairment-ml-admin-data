# fmt: off
"""
POST-HOC ANALYSIS: Demographics of Misclassified Patients

For each classification task, the best-performing model (by AUC-ROC) is
loaded from the saved primary analysis results (joblib), and its predictions
on the held-out test set are compared against the true labels. Patients are
grouped into "correctly classified" vs "misclassified", and demographic
characteristics are compared using statistical tests (t-tests for continuous,
chi-square/Fisher for categorical).

Usage:
    python scripts/run_posthoc_misclassification.py
"""

import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import collections
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import train_test_split
from joblib import load as joblib_load

from scripts.run_primary_analysis import (
    RANDOM_STATE, TEST_SET_SIZE,
    prepare_data_common,
    finalize_binary_data, finalize_multiclass_data,
    JaakkimainenBenchmark,  # Needed for joblib unpickling of saved results
)

# Register JaakkimainenBenchmark in __main__ so joblib/pickle can find it
# (The primary script saved models while running as __main__, so pickle
#  stored the class reference as __main__.JaakkimainenBenchmark)
import __main__
if not hasattr(__main__, 'JaakkimainenBenchmark'):
    __main__.JaakkimainenBenchmark = JaakkimainenBenchmark


# ============================================================================
# CONFIGURATION
# ============================================================================

# Demographic columns to analyze
DEMOGRAPHIC_COLS = {
    'age': {'type': 'continuous', 'label': 'Age (years)'},
    'sex': {'type': 'binary', 'label': 'Sex (1=Male)', 'categories': {0: 'Female', 1: 'Male'}},
    'education_years': {'type': 'continuous', 'label': 'Education (years)'},
    'race': {'type': 'categorical', 'label': 'Race/Ethnicity'},
    'mmsetotal': {'type': 'continuous', 'label': 'MMSE Total Score'},
    'moca_total': {'type': 'continuous', 'label': 'MoCA Total Score'},
}


def load_saved_models_and_data():
    """Load fitted models, test data, and feature names from saved joblib files.

    Returns:
        all_results: list of dicts, each with 'FinalModel', 'LabelDefinition', 'ModelName', etc.
        task_data: dict mapping task names to (X_train, y_train, X_test, y_test, orig_labels, Z_test)
        feature_names: dict mapping task names to list of feature name strings
    """
    models_dir = os.path.join(PROJECT_ROOT, "output", "primary", "trained_models")

    all_results_path = os.path.join(models_dir, "all_results.joblib")
    task_data_path = os.path.join(models_dir, "task_data.joblib")
    feature_names_path = os.path.join(models_dir, "feature_names.joblib")

    if not os.path.exists(all_results_path):
        print(f"  CRITICAL ERROR: {all_results_path} not found.")
        print(f"  Run the primary analysis first to generate saved models.")
        return None, None, None

    if not os.path.exists(task_data_path):
        print(f"  CRITICAL ERROR: {task_data_path} not found.")
        return None, None, None

    print(f"  Loading saved models from: {models_dir}")
    all_results = joblib_load(all_results_path)
    task_data = joblib_load(task_data_path)
    print(f"  Loaded {len(all_results)} model results across {len(task_data)} tasks")

    feature_names = None
    if os.path.exists(feature_names_path):
        feature_names = joblib_load(feature_names_path)

    return all_results, task_data, feature_names


def find_best_model_per_task(all_results):
    """From the list of all model results, find the best model per task by AUC.

    Returns a dict: {task_name: result_dict} where result_dict contains
    'FinalModel', 'ModelName', 'LabelDefinition', etc.
    """
    # Group results by task
    tasks = {}
    for r in all_results:
        task_name = r.get("LabelDefinition", "")
        if not task_name:
            continue
        # Skip Jaakkimainen benchmark and errored results
        if "Jaakkimainen" in r.get("ModelName", ""):
            continue
        if r.get("Error"):
            continue
        if r.get("FinalModel") is None:
            continue
        if task_name not in tasks:
            tasks[task_name] = []
        tasks[task_name].append(r)

    best_per_task = {}
    for task_name, results in tasks.items():
        best_result = None
        best_auc = -1
        for r in results:
            # Get AUC from point estimate metrics
            point_metrics = r.get("Test_Metrics_Point", {})
            auc_val = point_metrics.get("AUC", point_metrics.get("auc", -1))
            if auc_val is None:
                auc_val = -1
            try:
                auc_val = float(auc_val)
            except (ValueError, TypeError):
                auc_val = -1
            if auc_val > best_auc:
                best_auc = auc_val
                best_result = r

        if best_result is not None:
            best_per_task[task_name] = best_result
            print(f"    {task_name}: {best_result['ModelName']} (AUC={best_auc:.3f})")

    return best_per_task


def compare_demographics(correct_df, misclassified_df, output_path=None):
    """Compare demographics between correctly and incorrectly classified patients.

    Returns a DataFrame with test statistics and p-values.
    """
    results = []

    for col_name, col_info in DEMOGRAPHIC_COLS.items():
        if col_name not in correct_df.columns or col_name not in misclassified_df.columns:
            continue

        correct_vals = correct_df[col_name].dropna()
        misclass_vals = misclassified_df[col_name].dropna()

        if len(correct_vals) == 0 or len(misclass_vals) == 0:
            continue

        row = {
            'Variable': col_info['label'],
            'Correct_N': len(correct_vals),
            'Misclassified_N': len(misclass_vals),
        }

        if col_info['type'] == 'continuous':
            row['Correct_Mean'] = correct_vals.mean()
            row['Correct_SD'] = correct_vals.std()
            row['Misclassified_Mean'] = misclass_vals.mean()
            row['Misclassified_SD'] = misclass_vals.std()
            row['Correct_Summary'] = f"{correct_vals.mean():.1f} ({correct_vals.std():.1f})"
            row['Misclassified_Summary'] = f"{misclass_vals.mean():.1f} ({misclass_vals.std():.1f})"

            # Independent samples t-test (Welch's)
            if correct_vals.std() > 0 or misclass_vals.std() > 0:
                t_stat, p_val = stats.ttest_ind(correct_vals, misclass_vals, equal_var=False)
                row['Test'] = "Welch's t-test"
                row['Statistic'] = t_stat
                row['P_value'] = p_val
            else:
                row['Test'] = 'N/A (no variance)'
                row['Statistic'] = np.nan
                row['P_value'] = np.nan

        elif col_info['type'] in ('binary', 'categorical'):
            # Frequency table
            all_categories = sorted(set(correct_vals.unique()) | set(misclass_vals.unique()))

            correct_counts = correct_vals.value_counts()
            misclass_counts = misclass_vals.value_counts()

            # Summary as count (%)
            correct_parts = []
            misclass_parts = []
            for cat in all_categories:
                c_n = correct_counts.get(cat, 0)
                m_n = misclass_counts.get(cat, 0)
                c_pct = 100 * c_n / len(correct_vals) if len(correct_vals) > 0 else 0
                m_pct = 100 * m_n / len(misclass_vals) if len(misclass_vals) > 0 else 0
                cat_label = col_info.get('categories', {}).get(cat, str(cat))
                correct_parts.append(f"{cat_label}: {c_n} ({c_pct:.1f}%)")
                misclass_parts.append(f"{cat_label}: {m_n} ({m_pct:.1f}%)")

            row['Correct_Summary'] = '; '.join(correct_parts)
            row['Misclassified_Summary'] = '; '.join(misclass_parts)

            # Chi-squared test
            contingency = pd.DataFrame({
                'Correct': [correct_counts.get(cat, 0) for cat in all_categories],
                'Misclassified': [misclass_counts.get(cat, 0) for cat in all_categories]
            }, index=all_categories)

            # Remove rows where both columns are 0
            contingency = contingency[(contingency > 0).any(axis=1)]

            if contingency.shape[0] >= 2:
                # Use Fisher's exact test for 2x2, chi-squared otherwise
                if contingency.shape[0] == 2 and contingency.shape[1] == 2:
                    odds_ratio, p_val = stats.fisher_exact(contingency.values)
                    row['Test'] = "Fisher's exact test"
                    row['Statistic'] = odds_ratio
                    row['P_value'] = p_val
                else:
                    chi2, p_val, dof, expected = stats.chi2_contingency(contingency.values)
                    row['Test'] = f"Chi-squared (df={dof})"
                    row['Statistic'] = chi2
                    row['P_value'] = p_val
            else:
                row['Test'] = 'N/A (insufficient categories)'
                row['Statistic'] = np.nan
                row['P_value'] = np.nan

        results.append(row)

    results_df = pd.DataFrame(results)

    # Add significance markers
    if 'P_value' in results_df.columns:
        results_df['Significant'] = results_df['P_value'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            if pd.notna(p) else ''
        )

    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"    Saved: {output_path}")

    return results_df


def generate_demographics_figure(all_task_results, output_dir):
    """Generate a summary figure showing age/sex distributions for misclassified patients."""
    n_tasks = len(all_task_results)
    if n_tasks == 0:
        return

    fig, axes = plt.subplots(n_tasks, 2, figsize=(12, 3.5 * n_tasks))
    if n_tasks == 1:
        axes = axes.reshape(1, -1)

    for i, (task_name, task_result) in enumerate(all_task_results.items()):
        correct_df = task_result['correct_demographics']
        misclass_df = task_result['misclassified_demographics']

        # Age distribution (left panel)
        ax_age = axes[i, 0]
        if 'age' in correct_df.columns and 'age' in misclass_df.columns:
            correct_age = correct_df['age'].dropna()
            misclass_age = misclass_df['age'].dropna()

            bins = np.linspace(
                min(correct_age.min(), misclass_age.min()) if len(misclass_age) > 0 else correct_age.min(),
                max(correct_age.max(), misclass_age.max()) if len(misclass_age) > 0 else correct_age.max(),
                20
            )
            ax_age.hist(correct_age, bins=bins, alpha=0.6, label=f'Correct (n={len(correct_age)})',
                        color='steelblue', density=True, edgecolor='white')
            if len(misclass_age) > 0:
                ax_age.hist(misclass_age, bins=bins, alpha=0.6, label=f'Misclassified (n={len(misclass_age)})',
                            color='indianred', density=True, edgecolor='white')
            ax_age.set_xlabel('Age (years)')
            ax_age.set_ylabel('Density')
            ax_age.legend(fontsize=8)

            # Clean task name for title
            clean_name = task_name.split('. ', 1)[-1] if '. ' in task_name else task_name
            ax_age.set_title(f'{clean_name}\nAge Distribution', fontsize=10)

        # Sex distribution (right panel)
        ax_sex = axes[i, 1]
        if 'sex' in correct_df.columns and 'sex' in misclass_df.columns:
            correct_sex = correct_df['sex'].dropna()
            misclass_sex = misclass_df['sex'].dropna()

            categories = ['Female', 'Male']
            correct_counts = [sum(correct_sex == 0), sum(correct_sex == 1)]
            misclass_counts = [sum(misclass_sex == 0), sum(misclass_sex == 1)]

            correct_pct = [100 * c / len(correct_sex) if len(correct_sex) > 0 else 0 for c in correct_counts]
            misclass_pct = [100 * c / len(misclass_sex) if len(misclass_sex) > 0 else 0 for c in misclass_counts]

            x = np.arange(len(categories))
            width = 0.35
            ax_sex.bar(x - width/2, correct_pct, width, label=f'Correct (n={len(correct_sex)})',
                       color='steelblue', edgecolor='white')
            ax_sex.bar(x + width/2, misclass_pct, width, label=f'Misclassified (n={len(misclass_sex)})',
                       color='indianred', edgecolor='white')
            ax_sex.set_xticks(x)
            ax_sex.set_xticklabels(categories)
            ax_sex.set_ylabel('Percentage (%)')
            ax_sex.legend(fontsize=8)

            clean_name = task_name.split('. ', 1)[-1] if '. ' in task_name else task_name
            ax_sex.set_title(f'{clean_name}\nSex Distribution', fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'misclassification_demographics_summary.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary figure: {fig_path}")

    # Also save as PDF
    fig_path_pdf = fig_path.replace('.png', '.pdf')
    fig, axes = plt.subplots(n_tasks, 2, figsize=(12, 3.5 * n_tasks))
    if n_tasks == 1:
        axes = axes.reshape(1, -1)
    for i, (task_name, task_result) in enumerate(all_task_results.items()):
        correct_df = task_result['correct_demographics']
        misclass_df = task_result['misclassified_demographics']
        ax_age = axes[i, 0]
        if 'age' in correct_df.columns and 'age' in misclass_df.columns:
            correct_age = correct_df['age'].dropna()
            misclass_age = misclass_df['age'].dropna()
            bins = np.linspace(
                min(correct_age.min(), misclass_age.min()) if len(misclass_age) > 0 else correct_age.min(),
                max(correct_age.max(), misclass_age.max()) if len(misclass_age) > 0 else correct_age.max(), 20)
            ax_age.hist(correct_age, bins=bins, alpha=0.6, label=f'Correct (n={len(correct_age)})', color='steelblue', density=True, edgecolor='white')
            if len(misclass_age) > 0:
                ax_age.hist(misclass_age, bins=bins, alpha=0.6, label=f'Misclassified (n={len(misclass_age)})', color='indianred', density=True, edgecolor='white')
            ax_age.set_xlabel('Age (years)')
            ax_age.set_ylabel('Density')
            ax_age.legend(fontsize=8)
            clean_name = task_name.split('. ', 1)[-1] if '. ' in task_name else task_name
            ax_age.set_title(f'{clean_name}\nAge Distribution', fontsize=10)
        ax_sex = axes[i, 1]
        if 'sex' in correct_df.columns and 'sex' in misclass_df.columns:
            correct_sex = correct_df['sex'].dropna()
            misclass_sex = misclass_df['sex'].dropna()
            categories = ['Female', 'Male']
            correct_counts = [sum(correct_sex == 0), sum(correct_sex == 1)]
            misclass_counts = [sum(misclass_sex == 0), sum(misclass_sex == 1)]
            correct_pct = [100 * c / len(correct_sex) if len(correct_sex) > 0 else 0 for c in correct_counts]
            misclass_pct = [100 * c / len(misclass_sex) if len(misclass_sex) > 0 else 0 for c in misclass_counts]
            x = np.arange(len(categories))
            width = 0.35
            ax_sex.bar(x - width/2, correct_pct, width, label=f'Correct (n={len(correct_sex)})', color='steelblue', edgecolor='white')
            ax_sex.bar(x + width/2, misclass_pct, width, label=f'Misclassified (n={len(misclass_sex)})', color='indianred', edgecolor='white')
            ax_sex.set_xticks(x)
            ax_sex.set_xticklabels(categories)
            ax_sex.set_ylabel('Percentage (%)')
            ax_sex.legend(fontsize=8)
            clean_name = task_name.split('. ', 1)[-1] if '. ' in task_name else task_name
            ax_sex.set_title(f'{clean_name}\nSex Distribution', fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary figure: {fig_path_pdf}")


def main():
    print("=" * 70)
    print("POST-HOC ANALYSIS: Demographics of Misclassified Patients")
    print("=" * 70)

    start_time = time.time()

    # ======================================================================
    # Step 1: Load saved models and test data from primary analysis
    # ======================================================================
    print("\n=== Loading Saved Models ===")
    all_results, task_data, feature_names = load_saved_models_and_data()
    if all_results is None or task_data is None:
        print("CRITICAL ERROR: Cannot proceed without saved models. Exiting.")
        return

    # Find the best model per task (by AUC)
    print("\n=== Identifying Best Model Per Task ===")
    best_per_task = find_best_model_per_task(all_results)

    if not best_per_task:
        print("CRITICAL ERROR: No valid models found in saved results. Exiting.")
        return

    # ======================================================================
    # Step 2: Load raw CSV for demographic lookup
    # ======================================================================
    data_dir = os.path.join(PROJECT_ROOT, "data")
    possible_data_paths = [
        os.path.join(data_dir, "MAIN_new.csv"),
        os.path.join(data_dir, "MAIN.csv"),
    ]
    data_path = None
    for p in possible_data_paths:
        if os.path.exists(p):
            data_path = p
            break
    if data_path is None:
        print(f"CRITICAL ERROR: CSV data file not found in {data_dir}")
        return

    # --- Output ---
    output_dir = os.path.join(PROJECT_ROOT, "output", "posthoc")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved in: {output_dir}")

    # --- Load Data ---
    print("\n=== Loading Data for Demographic Lookup ===")
    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    df['cognitive_status'] = df['cognitive_status'].fillna('unknown').astype(str)
    df['cognitive_status'] = df['cognitive_status'].str.lower().str.strip().str.replace(' ', '_', regex=False)

    # --- Exclusion columns (same as primary analysis) ---
    exclude_columns_for_X = sorted(list(set([
        "id", "prompt_id", "prompt_visitdate", "visit_year", "jaak_hosp_code", "dob",
        "dementia", "cognitive_label", "dem/norm/oth", "jaak_Dem", "cognitive_status",
        "jaak_rx_filled", "jaak_3claims+_2yr_30days",
        "race", "education_years", "education_level", "sex", "age",
        "moca_total", "mmsetotal",
        "living_arrangement",
    ])))

    # --- Task definitions (same as primary analysis) ---
    task_dfs = collections.OrderedDict()
    task_dfs["1. Def Normal vs. Def Dementia"] = df[df["cognitive_status"].isin(["definite_normal", "definite_dementia"])].copy()
    task_dfs["2. Def+Pos Normal vs. Def+Pos Dementia"] = df[df["cognitive_status"].isin(["definite_normal", "possible_normal", "definite_dementia", "possible_dementia"])].copy()
    task_dfs["3. Def Normal vs. Def MCI"] = df[df["cognitive_status"].isin(["definite_normal", "definite_mci"])].copy()
    task_dfs["4. Def+Pos Normal vs. Def+Pos MCI"] = df[df["cognitive_status"].isin(["definite_normal", "possible_normal", "definite_mci", "possible_mci"])].copy()
    task_dfs["5. Multi: Def Normal vs. Def MCI vs. Def Dementia"] = df[df["cognitive_status"].isin(["definite_normal", "definite_mci", "definite_dementia"])].copy()
    task_dfs["6. Multi: Def+Pos Normal vs. Def+Pos MCI vs. Def+Pos Dementia"] = df[df["cognitive_status"].isin(["definite_normal", "possible_normal", "definite_mci", "possible_mci", "definite_dementia", "possible_dementia"])].copy()

    # ======================================================================
    # Step 3: For each task, load model + data, predict, analyze demographics
    # ======================================================================
    all_task_results = {}
    summary_rows = []

    for task_name, df_subset in task_dfs.items():
        print(f"\n{'='*60}")
        print(f"TASK: {task_name}")
        print(f"{'='*60}")

        if df_subset.empty:
            print(f"  No data for this task, skipping.")
            continue

        # Check if we have a saved model for this task
        if task_name not in best_per_task:
            print(f"  No saved model found for this task, skipping.")
            continue

        best_result = best_per_task[task_name]
        model_name = best_result["ModelName"]
        fitted_model = best_result["FinalModel"]
        is_multi = "Multi:" in task_name

        # Check if we have saved test data for this task
        if task_name not in task_data:
            print(f"  No saved test data for this task, skipping.")
            continue

        # Load the saved test data (X_test, y_test are numpy arrays)
        saved = task_data[task_name]
        X_train_saved, y_train_saved, X_test_saved, y_test_saved = saved[0], saved[1], saved[2], saved[3]

        print(f"  Best model: {model_name}")
        print(f"  Test set: {X_test_saved.shape[0]} samples, {X_test_saved.shape[1]} features")

        # ---------------------------------------------------------------
        # Recover test set indices by re-splitting with same random seed
        # (This reproduces the exact same train/test split as primary)
        # ---------------------------------------------------------------
        result = prepare_data_common(df_subset, task_name, exclude_columns_for_X)
        if result[0] is None:
            print(f"  ERROR: Data preparation failed")
            continue
        X_raw_df, y_ser, _ = result

        if is_multi:
            X_final_df, y_final_np, all_orig_labels = finalize_multiclass_data(X_raw_df, y_ser)
            n_classes = 3
        else:
            pos_labels_str = []
            if "Dementia" in task_name and "Normal" in task_name:
                pos_labels_str = ["definite_dementia", "possible_dementia"]
            elif "MCI" in task_name and "Normal" in task_name:
                pos_labels_str = ["definite_mci", "possible_mci"]
            X_final_df, y_final_np = finalize_binary_data(X_raw_df, y_ser, pos_labels_str)
            n_classes = 2

        if X_final_df is None:
            print(f"  ERROR: Label finalization failed")
            continue

        # Re-split with identical parameters to recover test indices
        X_train_raw_df, X_test_raw_df, y_train_split, y_test_split = train_test_split(
            X_final_df, y_final_np,
            test_size=TEST_SET_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_final_np
        )

        # Verify the split matches the saved data
        if len(y_test_split) != len(y_test_saved):
            print(f"  WARNING: Test set size mismatch (split={len(y_test_split)}, saved={len(y_test_saved)})")
            print(f"  Falling back to saved data without demographic lookup.")
            continue

        test_indices = X_test_raw_df.index

        # ---------------------------------------------------------------
        # Use the SAVED fitted model to predict on SAVED test data
        # (This ensures predictions match the primary analysis exactly)
        # ---------------------------------------------------------------
        y_test = y_test_saved
        y_pred = fitted_model.predict(X_test_saved)

        # Identify correct vs misclassified
        correct_mask = (y_pred == y_test)
        misclass_mask = ~correct_mask

        n_correct = correct_mask.sum()
        n_misclass = misclass_mask.sum()
        accuracy = n_correct / len(y_test)
        print(f"  Correct: {n_correct} ({100*accuracy:.1f}%), Misclassified: {n_misclass} ({100*(1-accuracy):.1f}%)")

        if n_misclass == 0:
            print(f"  No misclassified patients — skipping demographic comparison.")
            continue

        # Look up demographics from original data using recovered test indices
        test_demographics = df_subset.loc[test_indices, list(DEMOGRAPHIC_COLS.keys())].copy()
        test_demographics = test_demographics.reset_index(drop=True)

        correct_demographics = test_demographics[correct_mask]
        misclass_demographics = test_demographics[misclass_mask]

        # Compare demographics
        print(f"\n  --- Demographic Comparison ---")
        safe_name = task_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
        csv_path = os.path.join(output_dir, f"demographics_{safe_name}.csv")

        comparison_df = compare_demographics(correct_demographics, misclass_demographics, csv_path)

        # Print results
        if not comparison_df.empty:
            for _, row in comparison_df.iterrows():
                sig = row.get('Significant', '')
                p_str = f"p={row['P_value']:.4f}" if pd.notna(row.get('P_value')) else 'N/A'
                print(f"    {row['Variable']:25s} | Correct: {row.get('Correct_Summary', 'N/A'):30s} | "
                      f"Misclassified: {row.get('Misclassified_Summary', 'N/A'):30s} | {p_str} {sig}")

        # Store for summary figure
        all_task_results[task_name] = {
            'correct_demographics': correct_demographics,
            'misclassified_demographics': misclass_demographics,
            'comparison': comparison_df,
            'n_correct': n_correct,
            'n_misclassified': n_misclass,
            'accuracy': accuracy,
        }

        # Summary row
        summary_rows.append({
            'Task': task_name,
            'Model': model_name,
            'N_Test': len(y_test),
            'N_Correct': n_correct,
            'N_Misclassified': n_misclass,
            'Accuracy': f"{accuracy:.3f}",
        })

        # Misclassification error breakdown for multiclass
        if is_multi:
            class_names = ['Normal', 'MCI', 'Dementia']
            print(f"\n  --- Error Breakdown (Multiclass) ---")
            for true_class in range(n_classes):
                for pred_class in range(n_classes):
                    if true_class == pred_class:
                        continue
                    mask = (y_test == true_class) & (y_pred == pred_class)
                    count = mask.sum()
                    if count > 0:
                        error_demographics = test_demographics[mask]
                        mean_age = error_demographics['age'].mean() if 'age' in error_demographics.columns else np.nan
                        pct_male = error_demographics['sex'].mean() * 100 if 'sex' in error_demographics.columns else np.nan
                        print(f"    {class_names[true_class]} -> {class_names[pred_class]}: "
                              f"n={count}, mean age={mean_age:.1f}, male={pct_male:.0f}%")

    # ======================================================================
    # Step 4: Generate summary outputs
    # ======================================================================
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, 'misclassification_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Saved summary: {summary_path}")

    if all_task_results:
        generate_demographics_figure(all_task_results, output_dir)

    # Combined demographics table across all tasks
    if all_task_results:
        combined_rows = []
        for task_name, task_result in all_task_results.items():
            comp = task_result['comparison']
            for _, row in comp.iterrows():
                combined_row = dict(row)
                combined_row['Task'] = task_name
                combined_rows.append(combined_row)

        if combined_rows:
            combined_df = pd.DataFrame(combined_rows)
            cols_order = ['Task'] + [c for c in combined_df.columns if c != 'Task']
            combined_df = combined_df[cols_order]
            combined_path = os.path.join(output_dir, 'all_tasks_demographics_comparison.csv')
            combined_df.to_csv(combined_path, index=False)
            print(f"  Saved combined table: {combined_path}")

    total_elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"POST-HOC ANALYSIS COMPLETE (Total time: {total_elapsed:.1f}s)")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
# fmt: on
