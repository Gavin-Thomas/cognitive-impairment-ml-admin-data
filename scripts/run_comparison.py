"""
Compare Primary Analysis (without age/sex) vs Secondary Analysis (with age/sex).

Generates side-by-side comparison tables showing how including age and sex
as predictors affects model performance across all classification tasks.

Usage:
    python compare_primary_secondary.py

Prerequisites:
    - Primary analysis (original_parallel.py) must have been run
    - Secondary analysis (secondary_analysis_age_sex.py) must have been run
"""

import os
import re
import sys
import glob
import numpy as np
import pandas as pd


# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRIMARY_DIR = os.path.join(PROJECT_ROOT, "output", "primary")
SECONDARY_DIR = os.path.join(PROJECT_ROOT, "output", "secondary")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "comparison")

# Metrics to compare (column name prefix -> short label)
METRICS_TO_COMPARE = [
    ("AUC", "AUC"),
    ("PR AUC", "PR AUC"),
    ("Sensitivity", "Sensitivity"),
    ("Specificity", "Specificity"),
    ("PPV", "PPV"),
    ("NPV", "NPV"),
    ("F1 Score", "F1"),
    ("Balanced Acc.", "Bal. Acc."),
]


def find_metrics_csv(base_dir):
    """Find the final evaluation metrics CSV in a results directory."""
    results_dir = os.path.join(base_dir, "results_final_eval")
    pattern = os.path.join(results_dir, "final_evaluation_metrics_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    # Return the most recently modified file
    return max(matches, key=os.path.getmtime)


def parse_point_value(cell_str):
    """Extract the point estimate from a cell like '0.943 [0.945 (0.882-0.973)]'."""
    if pd.isna(cell_str) or cell_str == "N/A":
        return np.nan
    cell_str = str(cell_str).strip()
    # Point value is the first number before the bracket
    match = re.match(r'^([\d.]+)', cell_str)
    if match:
        return float(match.group(1))
    return np.nan


def parse_ci(cell_str):
    """Extract bootstrap median and CI from a cell like '0.943 [0.945 (0.882-0.973)]'.

    Returns (bs_median, ci_low, ci_high) or (nan, nan, nan).
    """
    if pd.isna(cell_str) or cell_str == "N/A":
        return np.nan, np.nan, np.nan
    cell_str = str(cell_str).strip()
    match = re.search(r'\[([\d.]+)\s*\(([\d.]+)-([\d.]+)\)\]', cell_str)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    return np.nan, np.nan, np.nan


def load_metrics(csv_path):
    """Load metrics CSV and return a DataFrame indexed by (Task, Model)."""
    df = pd.read_csv(csv_path)
    df = df.set_index(["Task", "Model"])
    return df


def get_metric_column(df, metric_prefix):
    """Find the full column name matching a metric prefix."""
    for col in df.columns:
        if col.startswith(metric_prefix + " ("):
            return col
    return None


def create_comparison_table(primary_df, secondary_df):
    """Create a side-by-side comparison table with deltas.

    Returns a DataFrame with columns:
        Task, Model, Metric, Primary, Secondary, Delta, Primary_CI, Secondary_CI
    """
    rows = []

    # Get common (Task, Model) pairs
    common_idx = primary_df.index.intersection(secondary_df.index)

    for task, model in common_idx:
        for metric_prefix, metric_label in METRICS_TO_COMPARE:
            pri_col = get_metric_column(primary_df, metric_prefix)
            sec_col = get_metric_column(secondary_df, metric_prefix)

            if pri_col is None or sec_col is None:
                continue

            pri_cell = primary_df.loc[(task, model), pri_col]
            sec_cell = secondary_df.loc[(task, model), sec_col]

            pri_point = parse_point_value(pri_cell)
            sec_point = parse_point_value(sec_cell)
            pri_med, pri_lo, pri_hi = parse_ci(pri_cell)
            sec_med, sec_lo, sec_hi = parse_ci(sec_cell)

            delta = sec_point - pri_point if not (np.isnan(pri_point) or np.isnan(sec_point)) else np.nan

            # Check CI overlap (rough indicator of significance)
            ci_overlap = True
            if not any(np.isnan(x) for x in [pri_lo, pri_hi, sec_lo, sec_hi]):
                ci_overlap = not (sec_lo > pri_hi or pri_lo > sec_hi)

            rows.append({
                "Task": task,
                "Model": model,
                "Metric": metric_label,
                "Primary (no age/sex)": f"{pri_point:.3f}" if not np.isnan(pri_point) else "N/A",
                "Secondary (with age/sex)": f"{sec_point:.3f}" if not np.isnan(sec_point) else "N/A",
                "Delta": f"{delta:+.3f}" if not np.isnan(delta) else "N/A",
                "Primary CI": f"({pri_lo:.3f}-{pri_hi:.3f})" if not np.isnan(pri_lo) else "N/A",
                "Secondary CI": f"({sec_lo:.3f}-{sec_hi:.3f})" if not np.isnan(sec_lo) else "N/A",
                "CIs Overlap": "Yes" if ci_overlap else "No*",
                "_delta_numeric": delta,
            })

    return pd.DataFrame(rows)


def create_summary_table(comparison_df):
    """Create a summary table showing average deltas per task and per model.

    Focuses on AUC and Balanced Accuracy as the primary metrics.
    """
    summary_rows = []
    key_metrics = ["AUC", "Bal. Acc."]

    for metric in key_metrics:
        metric_df = comparison_df[comparison_df["Metric"] == metric].copy()

        # Per-task summary
        for task in metric_df["Task"].unique():
            task_df = metric_df[metric_df["Task"] == task]
            deltas = task_df["_delta_numeric"].dropna()
            non_overlap = (task_df["CIs Overlap"] == "No*").sum()
            total = len(task_df)

            summary_rows.append({
                "Metric": metric,
                "Group": task,
                "Mean Delta": f"{deltas.mean():+.3f}" if len(deltas) > 0 else "N/A",
                "Max Improvement": f"{deltas.max():+.3f}" if len(deltas) > 0 else "N/A",
                "Max Decline": f"{deltas.min():+.3f}" if len(deltas) > 0 else "N/A",
                "Models with Non-Overlapping CIs": f"{non_overlap}/{total}",
            })

    return pd.DataFrame(summary_rows)


def main():
    print("=" * 70)
    print("COMPARISON: Primary Analysis vs Secondary Analysis (Age/Sex)")
    print("=" * 70)

    # Find metrics CSVs
    primary_csv = find_metrics_csv(PRIMARY_DIR)
    secondary_csv = find_metrics_csv(SECONDARY_DIR)

    if primary_csv is None:
        print(f"\nERROR: Primary analysis results not found in: {PRIMARY_DIR}")
        print("Run original_parallel.py first.")
        sys.exit(1)

    if secondary_csv is None:
        print(f"\nERROR: Secondary analysis results not found in: {SECONDARY_DIR}")
        print("Run secondary_analysis_age_sex.py first.")
        sys.exit(1)

    print(f"\nPrimary results:   {primary_csv}")
    print(f"Secondary results: {secondary_csv}")

    # Load data
    primary_df = load_metrics(primary_csv)
    secondary_df = load_metrics(secondary_csv)

    print(f"\nPrimary:   {len(primary_df)} model-task combinations")
    print(f"Secondary: {len(secondary_df)} model-task combinations")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate comparison
    comparison_df = create_comparison_table(primary_df, secondary_df)

    if comparison_df.empty:
        print("\nERROR: No common model-task combinations found between analyses.")
        sys.exit(1)

    # Save full comparison
    full_path = os.path.join(OUTPUT_DIR, "full_comparison_primary_vs_secondary.csv")
    comparison_df.drop(columns=["_delta_numeric"]).to_csv(full_path, index=False)
    print(f"\nFull comparison saved to: {full_path}")

    # Per-metric comparison tables
    for metric_prefix, metric_label in METRICS_TO_COMPARE:
        metric_df = comparison_df[comparison_df["Metric"] == metric_label].copy()
        metric_df = metric_df.drop(columns=["_delta_numeric"])

        safe_name = metric_label.replace(" ", "_").replace(".", "")
        metric_path = os.path.join(OUTPUT_DIR, f"comparison_{safe_name}.csv")
        metric_df.to_csv(metric_path, index=False)

    # Summary table
    summary_df = create_summary_table(comparison_df)
    summary_path = os.path.join(OUTPUT_DIR, "summary_impact_of_age_sex.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY: Impact of Including Age and Sex as Predictors")
    print("=" * 70)

    for metric in ["AUC", "Bal. Acc."]:
        metric_data = comparison_df[comparison_df["Metric"] == metric]
        deltas = metric_data["_delta_numeric"].dropna()
        non_overlap = (metric_data["CIs Overlap"] == "No*").sum()

        print(f"\n{metric}:")
        print(f"  Mean delta:    {deltas.mean():+.4f}")
        print(f"  Median delta:  {deltas.median():+.4f}")
        print(f"  Range:         [{deltas.min():+.4f}, {deltas.max():+.4f}]")
        print(f"  Non-overlapping CIs: {non_overlap}/{len(deltas)} comparisons")

        # Per-task breakdown
        print(f"\n  Per-task mean delta:")
        for task in sorted(metric_data["Task"].unique()):
            task_deltas = metric_data[metric_data["Task"] == task]["_delta_numeric"].dropna()
            if len(task_deltas) > 0:
                print(f"    {task}: {task_deltas.mean():+.4f}")

    # Best model comparison (best AUC per task)
    print("\n" + "=" * 70)
    print("BEST MODEL PER TASK: Primary vs Secondary (by AUC)")
    print("=" * 70)

    auc_df = comparison_df[comparison_df["Metric"] == "AUC"].copy()

    for task in sorted(auc_df["Task"].unique()):
        task_data = auc_df[auc_df["Task"] == task]

        # Find best model in primary
        best_pri_idx = task_data["Primary (no age/sex)"].apply(
            lambda x: float(x) if x != "N/A" else -1
        ).idxmax()
        best_pri = task_data.loc[best_pri_idx]

        # Find best model in secondary
        best_sec_idx = task_data["Secondary (with age/sex)"].apply(
            lambda x: float(x) if x != "N/A" else -1
        ).idxmax()
        best_sec = task_data.loc[best_sec_idx]

        print(f"\n{task}:")
        print(f"  Primary best:   {best_pri['Model']} (AUC={best_pri['Primary (no age/sex)']})")
        print(f"  Secondary best: {best_sec['Model']} (AUC={best_sec['Secondary (with age/sex)']})")

        if best_pri['Model'] != best_sec['Model']:
            print(f"  ** Best model changed when age/sex included")

    print(f"\n\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
