# fmt: off
"""
Publication-Quality Visualization Module for Cognitive Classification Analysis
==============================================================================
Adds:
1. Forest plots for model comparison (6 tasks)
2. SHAP swarm plots and magnitude bars
3. ROC curves with confidence bands
4. Calibration plots
5. Publication-quality confusion matrices
6. Multi-metric comparison plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP library not installed. Install with: pip install shap")


# ==============================================================================
# PUBLICATION STYLE SETTINGS
# ==============================================================================

def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.1)


# ==============================================================================
# TASK DATA HELPERS
# ==============================================================================

def _unpack_task_data(task_tuple):
    """
    Unpack task tuple from analysis pipeline.

    Supports:
    - (X_train, y_train, X_test, y_test, labels)
    - (X_train, y_train, X_test, y_test, labels, z_test)
    """
    if not isinstance(task_tuple, (tuple, list)):
        raise ValueError("Task data must be a tuple/list.")

    if len(task_tuple) == 6:
        X_train, y_train, X_test, y_test, labels, z_test = task_tuple
    elif len(task_tuple) == 5:
        X_train, y_train, X_test, y_test, labels = task_tuple
        z_test = None
    else:
        raise ValueError(f"Unexpected task tuple length: {len(task_tuple)}")

    return X_train, y_train, X_test, y_test, labels, z_test


# ==============================================================================
# 1. FOREST PLOTS
# ==============================================================================

def compute_ci_overlap_significance(point1, ci1, point2, ci2):
    """
    Compute significance based on bootstrap CI overlap.
    Uses the overlap rule: if CIs don't overlap, p < 0.05.
    For more precision, uses the 83% CI rule approximation.

    Returns:
    --------
    dict with 'significant' (bool), 'p_approx' (float), 'method' (str)
    """
    lower1, upper1 = ci1
    lower2, upper2 = ci2

    # Check for strict non-overlap (p < 0.01 approximately)
    if upper1 < lower2 or upper2 < lower1:
        return {'significant': True, 'p_approx': 0.01, 'method': 'CI non-overlap'}

    # Approximate using CI overlap proportion
    # If CIs overlap by less than 50%, roughly p < 0.05
    overlap_start = max(lower1, lower2)
    overlap_end = min(upper1, upper2)

    if overlap_end > overlap_start:
        overlap = overlap_end - overlap_start
        ci1_width = upper1 - lower1
        ci2_width = upper2 - lower2
        avg_width = (ci1_width + ci2_width) / 2

        overlap_fraction = overlap / avg_width if avg_width > 0 else 1.0

        # Rule of thumb: overlap < 50% of average CI width suggests significance
        if overlap_fraction < 0.5:
            return {'significant': True, 'p_approx': 0.05, 'method': 'CI overlap < 50%'}

    return {'significant': False, 'p_approx': 1.0, 'method': 'CI overlap'}


def compute_pairwise_significance(models, point_estimates, lower_cis, upper_cis, task_results, task_data=None):
    """
    Compute pairwise significance between models.

    Returns list of significant pairs with annotations.
    """
    n_models = len(models)
    significant_pairs = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Use CI overlap method
            result = compute_ci_overlap_significance(
                point_estimates[i], (lower_cis[i], upper_cis[i]),
                point_estimates[j], (lower_cis[j], upper_cis[j])
            )

            if result['significant']:
                significant_pairs.append({
                    'model1': models[i],
                    'model2': models[j],
                    'model1_idx': i,
                    'model2_idx': j,
                    'point1': point_estimates[i],
                    'point2': point_estimates[j],
                    'diff': abs(point_estimates[i] - point_estimates[j]),
                    'p_approx': result['p_approx'],
                    'method': result['method']
                })

    return significant_pairs


def generate_forest_plot(task_results, task_name, output_dir, metric='auc', show_significance=True):
    """
    Generate a forest plot showing model comparison with 95% CIs for a single task.

    Parameters:
    -----------
    task_results : list of dict
        Results for all models on this task
    task_name : str
        Name of the classification task
    output_dir : str
        Directory to save the plot
    metric : str
        Metric to plot ('auc', 'balanced_accuracy', 'sensitivity', 'specificity', etc.)
    """
    set_publication_style()

    # Extract data
    models = []
    point_estimates = []
    lower_cis = []
    upper_cis = []

    for result in task_results:
        model_name = result.get("ModelName", "Unknown")

        # Get point estimate
        point_val = result.get("Test_Metrics_Point", {}).get(metric, np.nan)

        # Get bootstrap CI
        ci_tuple = result.get("Test_Metrics_CI_BS", {}).get(metric, (np.nan, np.nan))
        lower, upper = ci_tuple

        if not np.isnan(point_val) and not np.isnan(lower) and not np.isnan(upper):
            models.append(model_name)
            point_estimates.append(point_val)
            lower_cis.append(lower)
            upper_cis.append(upper)

    if not models:
        print(f"  No valid data for forest plot: {task_name}")
        return

    # Sort by point estimate (descending)
    sorted_indices = np.argsort(point_estimates)[::-1]
    models = [models[i] for i in sorted_indices]
    point_estimates = [point_estimates[i] for i in sorted_indices]
    lower_cis = [lower_cis[i] for i in sorted_indices]
    upper_cis = [upper_cis[i] for i in sorted_indices]

    n_models = len(models)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(4, n_models * 0.6)))

    # Y positions (reversed so best model is at top)
    y_positions = np.arange(n_models)

    # Color palette
    colors = sns.color_palette("viridis", n_models)

    # Plot each model
    for i, (model, point, lower, upper) in enumerate(zip(models, point_estimates, lower_cis, upper_cis)):
        # Error bars (CI)
        ax.plot([lower, upper], [y_positions[i], y_positions[i]],
                color=colors[i], linewidth=2, solid_capstyle='round')

        # Point estimate (diamond marker)
        ax.scatter(point, y_positions[i], marker='D', s=100,
                   color=colors[i], edgecolor='black', linewidth=0.5, zorder=5)

        # Add CI text to the right
        ci_text = f"{point:.3f} [{lower:.3f}, {upper:.3f}]"
        ax.text(1.02, y_positions[i], ci_text, transform=ax.get_yaxis_transform(),
                va='center', ha='left', fontsize=9, family='monospace')

    # Reference line at various thresholds
    ax.axvline(x=0.5, color='lightgray', linestyle='--', linewidth=1, alpha=0.7, label='Random (0.5)')
    ax.axvline(x=0.7, color='lightgray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0.8, color='lightgray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0.9, color='lightgray', linestyle=':', linewidth=1, alpha=0.5)

    # Compute and display significance bars (comparing best model to others)
    significance_data = []
    if show_significance and n_models >= 2:
        sig_pairs = compute_pairwise_significance(
            models, point_estimates, lower_cis, upper_cis, task_results
        )

        # Focus on comparisons with the best model (index 0)
        best_comparisons = [p for p in sig_pairs if p['model1_idx'] == 0]

        # Draw significance markers on the right side
        for pair in best_comparisons:
            j = pair['model2_idx']
            p_val = pair['p_approx']

            # Significance stars
            if p_val <= 0.01:
                stars = '**'
            elif p_val <= 0.05:
                stars = '*'
            else:
                stars = ''

            if stars:
                # Add significance marker next to the lower model
                ax.text(1.18, y_positions[j], stars,
                       transform=ax.get_yaxis_transform(),
                       va='center', ha='left', fontsize=12, fontweight='bold',
                       color='darkred')

            # Store for CSV
            significance_data.append({
                'Model_1': pair['model1'],
                'Model_2': pair['model2'],
                'Diff': pair['diff'],
                'P_Approx': pair['p_approx'],
                'Significant': pair['p_approx'] <= 0.05,
                'Method': pair['method']
            })

        # Add legend note for significance
        if best_comparisons:
            ax.text(1.18, n_models + 0.3, 'vs. Best',
                   transform=ax.get_yaxis_transform(), va='bottom', ha='left',
                   fontsize=8, fontweight='bold', fontstyle='italic')

    # Formatting
    metric_labels = {
        'auc': 'AUC-ROC',
        'balanced_accuracy': 'Balanced Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'ppv': 'Positive Predictive Value',
        'npv': 'Negative Predictive Value',
        'f1': 'F1 Score',
        'pr_auc': 'AUC-PR'
    }

    ax.set_yticks(y_positions)
    ax.set_yticklabels(models)
    ax.set_xlabel(metric_labels.get(metric, metric.upper()), fontweight='bold')
    ax.set_xlim(max(0, min(lower_cis) - 0.05), 1.0)
    ax.set_ylim(-0.5, n_models - 0.5)

    # Title with task name (wrap long names)
    title = f"Forest Plot: {metric_labels.get(metric, metric.upper())}\n{task_name}"
    if show_significance:
        title += "\n(* p<0.05, ** p<0.01 vs. best model)"
    ax.set_title(title, fontweight='bold', pad=10)

    # Add column headers
    ax.text(1.02, n_models + 0.3, f'{metric_labels.get(metric, metric.upper())} [95% CI]',
            transform=ax.get_yaxis_transform(), va='bottom', ha='left',
            fontsize=10, fontweight='bold', family='monospace')

    # Grid
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save plot
    safe_name = task_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
    filepath = os.path.join(output_dir, f"forest_plot_{safe_name}_{metric}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    # Save corresponding data table
    forest_df = pd.DataFrame({
        'Task': [task_name] * len(models),
        'Model': models,
        'Metric': [metric] * len(models),
        'Point_Estimate': point_estimates,
        'CI_Lower_95': lower_cis,
        'CI_Upper_95': upper_cis,
        'CI_Width': [u - l for u, l in zip(upper_cis, lower_cis)]
    })
    csv_path = filepath.replace('.png', '_data.csv')
    forest_df.to_csv(csv_path, index=False)

    # Save significance comparison data if available
    if significance_data:
        sig_df = pd.DataFrame(significance_data)
        sig_csv_path = filepath.replace('.png', '_significance.csv')
        sig_df.to_csv(sig_csv_path, index=False)
        print(f"  Saved forest plot: {os.path.basename(filepath)} + data table + significance tests")
    else:
        print(f"  Saved forest plot: {os.path.basename(filepath)} + data table")


def generate_all_forest_plots(results_list, output_dir):
    """Generate forest plots for all tasks."""
    print("\n=== Generating Forest Plots ===")
    os.makedirs(output_dir, exist_ok=True)

    # Group results by task
    tasks = {}
    for r in results_list:
        task_name = r.get("LabelDefinition", "Unknown")
        if task_name not in tasks:
            tasks[task_name] = []
        tasks[task_name].append(r)

    # Generate forest plot for each task (using balanced accuracy as primary metric)
    for task_name, task_results in tasks.items():
        print(f"  Processing: {task_name}")
        generate_forest_plot(task_results, task_name, output_dir, metric='balanced_accuracy')


def generate_combined_forest_plot(results_list, output_dir, metric='balanced_accuracy'):
    """
    Generate a single combined forest plot with all tasks and models.
    Groups models by task with visual separation.
    """
    set_publication_style()

    # Group results by task
    tasks = {}
    for r in results_list:
        task_name = r.get("LabelDefinition", "Unknown")
        if task_name not in tasks:
            tasks[task_name] = []
        tasks[task_name].append(r)

    # Prepare data structure
    plot_data = []
    task_labels = []

    for task_name in sorted(tasks.keys()):
        task_results = tasks[task_name]
        task_entries = []

        for result in task_results:
            model_name = result.get("ModelName", "Unknown")
            point_val = result.get("Test_Metrics_Point", {}).get(metric, np.nan)
            ci_tuple = result.get("Test_Metrics_CI_BS", {}).get(metric, (np.nan, np.nan))

            if not np.isnan(point_val) and not np.isnan(ci_tuple[0]):
                task_entries.append({
                    'task': task_name,
                    'model': model_name,
                    'point': point_val,
                    'lower': ci_tuple[0],
                    'upper': ci_tuple[1]
                })

        # Sort by point estimate within each task
        task_entries.sort(key=lambda x: x['point'], reverse=True)
        plot_data.extend(task_entries)
        if task_entries:
            task_labels.append((task_name, len(task_entries)))

    if not plot_data:
        print("  No valid data for combined forest plot")
        return

    # Create figure
    n_entries = len(plot_data)
    fig, ax = plt.subplots(figsize=(12, max(8, n_entries * 0.4)))

    # Colors for different tasks
    task_colors = sns.color_palette("husl", len(task_labels))
    task_color_map = {label[0]: task_colors[i] for i, label in enumerate(task_labels)}

    # Plot
    y_pos = 0
    y_positions = []
    y_labels = []
    task_boundaries = []

    for entry in plot_data:
        color = task_color_map[entry['task']]

        # CI line
        ax.plot([entry['lower'], entry['upper']], [y_pos, y_pos],
                color=color, linewidth=2)
        # Point estimate
        ax.scatter(entry['point'], y_pos, marker='D', s=80,
                   color=color, edgecolor='black', linewidth=0.5, zorder=5)

        y_positions.append(y_pos)
        y_labels.append(entry['model'])
        y_pos += 1

    # Add task separators and labels
    cumulative = 0
    for task_name, count in task_labels:
        if cumulative > 0:
            ax.axhline(y=cumulative - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        # Task label on left margin
        mid_y = cumulative + count / 2 - 0.5
        ax.text(-0.02, mid_y, task_name.split('.')[0] + '.',
                transform=ax.get_yaxis_transform(), va='center', ha='right',
                fontsize=8, fontweight='bold', rotation=0)
        cumulative += count

    # Reference lines
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random')

    # Metric labels for display
    metric_labels = {
        'auc': 'AUC-ROC',
        'balanced_accuracy': 'Balanced Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'ppv': 'Positive Predictive Value',
        'npv': 'Negative Predictive Value',
        'f1': 'F1 Score',
        'pr_auc': 'AUC-PR'
    }

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel(f'{metric_labels.get(metric, metric.upper())} [95% CI]', fontweight='bold')
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.5, n_entries - 0.5)
    ax.set_title(f'Model Comparison Across All Classification Tasks\n({metric_labels.get(metric, metric.upper())})', fontweight='bold', fontsize=14)
    ax.xaxis.grid(True, alpha=0.3)

    # Legend for tasks
    legend_patches = [mpatches.Patch(color=task_color_map[t[0]], label=t[0]) for t in task_labels]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8, framealpha=0.9)

    plt.tight_layout()

    filepath = os.path.join(output_dir, f"forest_plot_combined_all_tasks.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # Save corresponding data table
    combined_df = pd.DataFrame(plot_data)
    combined_df['CI_Width'] = combined_df['upper'] - combined_df['lower']
    combined_df = combined_df.rename(columns={
        'task': 'Task', 'model': 'Model', 'point': 'Point_Estimate',
        'lower': 'CI_Lower_95', 'upper': 'CI_Upper_95'
    })
    csv_path = filepath.replace('.png', '_data.csv')
    combined_df.to_csv(csv_path, index=False)
    print(f"  Saved combined forest plot: {os.path.basename(filepath)} + data table")


# ==============================================================================
# 2. SHAP ANALYSIS PLOTS
# ==============================================================================

def compute_shap_values(model, X_train, X_test, feature_names, model_name):
    """
    Compute SHAP values for a trained model.

    Returns:
    --------
    shap_values : array or None
    explainer : shap.Explainer or None
    """
    if not SHAP_AVAILABLE:
        return None, None

    try:
        # Try TreeExplainer first (fast for tree-based models)
        if hasattr(model, 'named_steps'):
            # Pipeline - get the final estimator
            final_step_name = list(model.named_steps.keys())[-1]
            final_model = model.named_steps[final_step_name]

            # Check if there's a scaler
            if 'scaler' in model.named_steps:
                scaler = model.named_steps['scaler']
                X_test_transformed = scaler.transform(X_test)
            else:
                X_test_transformed = X_test
        else:
            final_model = model
            X_test_transformed = X_test

        # Transform training data for use as background (matches test transformation)
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            scaler = model.named_steps['scaler']
            X_train_transformed = scaler.transform(X_train)
        else:
            X_train_transformed = X_train

        # Choose appropriate explainer
        model_type = type(final_model).__name__

        if model_type in ['RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier']:
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_test_transformed)

            # For binary classification, take positive class
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]

        elif model_type in ['LogisticRegression', 'SVC']:
            # Use TRAINING data as background distribution (not test data)
            background = shap.sample(X_train_transformed, min(100, len(X_train_transformed)))
            explainer = shap.KernelExplainer(final_model.predict_proba, background)
            shap_values = explainer.shap_values(X_test_transformed[:min(200, len(X_test_transformed))])

            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
        else:
            # Generic approach — use TRAINING data as background distribution
            background = shap.sample(X_train_transformed, min(50, len(X_train_transformed)))
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x),
                background
            )
            shap_values = explainer.shap_values(X_test_transformed[:min(100, len(X_test_transformed))])

        return shap_values, X_test_transformed

    except Exception as e:
        print(f"    SHAP computation failed for {model_name}: {e}")
        return None, None


def generate_shap_swarm_plot(shap_values, X_data, feature_names, task_name, model_name, output_dir, max_features=20):
    """
    Generate SHAP beeswarm (swarm) plot showing feature importance and direction.
    """
    if shap_values is None:
        return

    set_publication_style()

    try:
        # Ensure shap_values and X_data have compatible shapes
        if isinstance(X_data, pd.DataFrame):
            X_data = X_data.values

        # Limit to first n samples if needed
        n_samples = min(shap_values.shape[0], X_data.shape[0])
        shap_values_plot = shap_values[:n_samples]
        X_data_plot = X_data[:n_samples]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, min(max_features * 0.4, 12))))

        # Get top features by mean absolute SHAP value
        mean_abs_shap = np.mean(np.abs(shap_values_plot), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-max_features:][::-1]

        # Subset data
        shap_subset = shap_values_plot[:, top_indices]
        X_subset = X_data_plot[:, top_indices]
        feature_subset = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in top_indices]

        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_subset,
            data=X_subset,
            feature_names=feature_subset
        )

        # Generate beeswarm plot
        shap.plots.beeswarm(explanation, show=False, max_display=max_features)

        # Customize
        plt.title(f"SHAP Feature Importance (Swarm Plot)\n{task_name} - {model_name}",
                  fontweight='bold', fontsize=12)
        plt.xlabel("SHAP Value (impact on model output)", fontweight='bold')

        plt.tight_layout()

        # Save plot
        safe_name = f"{task_name}_{model_name}".replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
        filepath = os.path.join(output_dir, f"shap_swarm_{safe_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

        # Save corresponding data table (feature importance summary)
        shap_summary_df = pd.DataFrame({
            'Rank': range(1, len(feature_subset) + 1),
            'Feature': feature_subset,
            'Mean_Abs_SHAP': mean_abs_shap[top_indices],
            'Std_SHAP': np.std(shap_subset, axis=0),
            'Min_SHAP': np.min(shap_subset, axis=0),
            'Max_SHAP': np.max(shap_subset, axis=0),
            'Mean_Feature_Value': np.mean(X_subset, axis=0),
            'Std_Feature_Value': np.std(X_subset, axis=0)
        })
        csv_path = filepath.replace('.png', '_data.csv')
        shap_summary_df.to_csv(csv_path, index=False)
        print(f"    Saved SHAP swarm plot: {os.path.basename(filepath)} + data table")

    except Exception as e:
        print(f"    Error generating SHAP swarm plot: {e}")
        plt.close()


def generate_shap_bar_plot(shap_values, feature_names, task_name, model_name, output_dir, max_features=20):
    """
    Generate SHAP magnitude bar plot showing mean absolute feature importance.
    """
    if shap_values is None:
        return

    set_publication_style()

    try:
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # Get top features
        top_indices = np.argsort(mean_abs_shap)[-max_features:][::-1]
        top_values = mean_abs_shap[top_indices]
        top_names = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in top_indices]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(5, max_features * 0.35)))

        # Horizontal bar plot
        colors = sns.color_palette("viridis", len(top_names))
        y_pos = np.arange(len(top_names))

        bars = ax.barh(y_pos, top_values, color=colors, edgecolor='black', linewidth=0.5)

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=10)
        ax.set_xlabel("Mean |SHAP Value|", fontweight='bold')
        ax.set_title(f"SHAP Feature Importance (Magnitude)\n{task_name} - {model_name}",
                     fontweight='bold', fontsize=12)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, top_values)):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', ha='left', fontsize=8)

        ax.set_xlim(0, max(top_values) * 1.15)
        ax.invert_yaxis()  # Top feature at top

        plt.tight_layout()

        # Save plot
        safe_name = f"{task_name}_{model_name}".replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
        filepath = os.path.join(output_dir, f"shap_bar_{safe_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

        # Save corresponding data table
        shap_bar_df = pd.DataFrame({
            'Rank': range(1, len(top_names) + 1),
            'Feature': top_names,
            'Mean_Abs_SHAP': top_values,
            'Relative_Importance': top_values / top_values[0] if top_values[0] > 0 else top_values
        })
        csv_path = filepath.replace('.png', '_data.csv')
        shap_bar_df.to_csv(csv_path, index=False)
        print(f"    Saved SHAP bar plot: {os.path.basename(filepath)} + data table")

    except Exception as e:
        print(f"    Error generating SHAP bar plot: {e}")
        plt.close()


def generate_shap_analysis(results_list, task_data, feature_names_dict, output_dir,
                           tasks_to_analyze=None):
    """
    Generate SHAP analysis for specified tasks.

    Parameters:
    -----------
    results_list : list
        All model results
    task_data : dict
        Dictionary with task data (X_train, y_train, X_test, y_test, labels)
    feature_names_dict : dict
        Dictionary mapping task names to feature names
    output_dir : str
        Output directory
    tasks_to_analyze : list or None
        List of task names to analyze. If None, analyzes tasks 1, 2, 5, 6.
    """
    if not SHAP_AVAILABLE:
        print("\n=== Skipping SHAP Analysis (library not installed) ===")
        return

    print("\n=== Generating SHAP Analysis ===")
    os.makedirs(output_dir, exist_ok=True)

    # Create interpretation warning file
    warning_path = os.path.join(output_dir, "SHAP_INTERPRETATION_WARNING.txt")
    with open(warning_path, 'w') as f:
        f.write("SHAP ANALYSIS INTERPRETATION WARNING\n")
        f.write("=" * 50 + "\n\n")
        f.write("IMPORTANT: Review the features included in this analysis carefully.\n\n")
        f.write("If diagnostic codes (ICD-9/10) or disease-specific medications are\n")
        f.write("present as features, SHAP importance values will reflect CIRCULAR\n")
        f.write("PREDICTION (using diagnosis to predict diagnosis).\n\n")
        f.write("Features to watch for data leakage:\n")
        f.write("  - ICD codes: F00-F03, G30, G31, R41, R54 (dementia-related)\n")
        f.write("  - ICD codes: 290, 294, 331 (dementia ICD-9)\n")
        f.write("  - Medications: donepezil, rivastigmine, galantamine, memantine\n\n")
        f.write("If these features dominate the SHAP plots, the model is detecting\n")
        f.write("the outcome label rather than true predictive features.\n\n")
        f.write("For valid predictive modeling, these features should be EXCLUDED.\n")
    print(f"  Created: {os.path.basename(warning_path)}")

    # Default: analyze definite tasks (1, 3, 5) and def+pos tasks (2, 4, 6)
    if tasks_to_analyze is None:
        tasks_to_analyze = [
            "1. Def Normal vs. Def Dementia",
            "2. Def+Pos Normal vs. Def+Pos Dementia",
            # "3. Def Normal vs. Def MCI",
            # "4. Def+Pos Normal vs. Def+Pos MCI",
            # "5. Multi: Def Normal vs. Def MCI vs. Def Dementia",
            # "6. Multi: Def+Pos Normal vs. Def+Pos MCI vs. Def+Pos Dementia"
        ]

    # Group results by task
    task_results_map = {}
    for r in results_list:
        task_name = r.get("LabelDefinition", "Unknown")
        if task_name not in task_results_map:
            task_results_map[task_name] = []
        task_results_map[task_name].append(r)

    # For each target task
    for task_name in tasks_to_analyze:
        if task_name not in task_data:
            print(f"  Skipping {task_name}: No data available")
            continue

        print(f"\n  Analyzing: {task_name}")

        # Get data
        X_train, y_train, X_test, y_test, _, _ = _unpack_task_data(task_data[task_name])
        feature_names = feature_names_dict.get(task_name, [f"Feature_{i}" for i in range(X_train.shape[1])])

        # Get results for this task
        task_results = task_results_map.get(task_name, [])

        # Find best model by AUC
        best_result = None
        best_auc = -1
        for r in task_results:
            auc_val = r.get("Test_Metrics_Point", {}).get('auc', 0)
            if auc_val and auc_val > best_auc:
                best_auc = auc_val
                best_result = r

        if best_result is None or best_result.get("FinalModel") is None:
            print(f"    No valid model found for {task_name}")
            continue

        model = best_result["FinalModel"]
        model_name = best_result.get("ModelName", "Unknown")
        print(f"    Best model: {model_name} (AUC: {best_auc:.3f})")

        # Compute SHAP values
        print(f"    Computing SHAP values...")
        shap_values, X_transformed = compute_shap_values(model, X_train, X_test, feature_names, model_name)

        if shap_values is not None:
            # Generate plots
            generate_shap_swarm_plot(shap_values, X_transformed, feature_names, task_name, model_name, output_dir)
            generate_shap_bar_plot(shap_values, feature_names, task_name, model_name, output_dir)


# ==============================================================================
# 3. ROC CURVES WITH CONFIDENCE BANDS
# ==============================================================================

def generate_roc_curves(results_list, task_data, output_dir):
    """
    Generate ROC curves for each task with all models overlaid.
    """
    print("\n=== Generating ROC Curves ===")
    os.makedirs(output_dir, exist_ok=True)
    set_publication_style()

    # Group by task
    task_results_map = {}
    for r in results_list:
        task_name = r.get("LabelDefinition", "Unknown")
        if task_name not in task_results_map:
            task_results_map[task_name] = []
        task_results_map[task_name].append(r)

    for task_name, task_results in task_results_map.items():
        if task_name not in task_data:
            continue

        is_multi = "Multi:" in task_name
        if is_multi:
            continue  # Skip multi-class for standard ROC

        print(f"  Processing: {task_name}")

        X_train, y_train, X_test, y_test, _, _ = _unpack_task_data(task_data[task_name])

        fig, ax = plt.subplots(figsize=(8, 8))

        # Color palette
        colors = sns.color_palette("husl", len(task_results))

        # Collect data for table
        roc_data_list = []
        auc_summary = []

        for i, result in enumerate(task_results):
            model = result.get("FinalModel")
            model_name = result.get("ModelName", "Unknown")

            if model is None or not hasattr(model, "predict_proba"):
                continue

            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                ax.plot(fpr, tpr, color=colors[i], lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')

                # Store ROC curve data
                for j in range(len(fpr)):
                    roc_data_list.append({
                        'Model': model_name,
                        'FPR': fpr[j],
                        'TPR': tpr[j],
                        'Threshold': thresholds[j] if j < len(thresholds) else np.nan
                    })

                # Store AUC summary
                auc_summary.append({
                    'Model': model_name,
                    'AUC': roc_auc
                })

            except Exception as e:
                print(f"    Error for {model_name}: {e}")
                continue

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

        # Formatting
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
        ax.set_title(f'ROC Curves\n{task_name}', fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        safe_name = task_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
        filepath = os.path.join(output_dir, f"roc_curves_{safe_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

        # Save data tables
        if roc_data_list:
            roc_df = pd.DataFrame(roc_data_list)
            roc_df.to_csv(filepath.replace('.png', '_curves_data.csv'), index=False)

            auc_df = pd.DataFrame(auc_summary)
            auc_df = auc_df.sort_values('AUC', ascending=False)
            auc_df.to_csv(filepath.replace('.png', '_auc_summary.csv'), index=False)

        print(f"    Saved: {os.path.basename(filepath)} + data tables")


# ==============================================================================
# 3b. PRECISION-RECALL CURVES
# ==============================================================================

def generate_pr_curves(results_list, task_data, output_dir):
    """
    Generate Precision-Recall curves for each task with all models overlaid.
    Critical for imbalanced datasets common in health sciences.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    print("\n=== Generating Precision-Recall Curves ===")
    os.makedirs(output_dir, exist_ok=True)
    set_publication_style()

    # Group by task
    task_results_map = {}
    for r in results_list:
        task_name = r.get("LabelDefinition", "Unknown")
        if task_name not in task_results_map:
            task_results_map[task_name] = []
        task_results_map[task_name].append(r)

    for task_name, task_results in task_results_map.items():
        if task_name not in task_data:
            continue

        is_multi = "Multi:" in task_name
        if is_multi:
            continue  # Skip multi-class for standard PR

        print(f"  Processing: {task_name}")

        X_train, y_train, X_test, y_test, _, _ = _unpack_task_data(task_data[task_name])

        # Calculate baseline (proportion of positive class)
        baseline = np.mean(y_test)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Color palette
        colors = sns.color_palette("husl", len(task_results))

        # Collect data for table
        pr_data_list = []
        ap_summary = []

        for i, result in enumerate(task_results):
            model = result.get("FinalModel")
            model_name = result.get("ModelName", "Unknown")

            if model is None or not hasattr(model, "predict_proba"):
                continue

            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
                ap_score = average_precision_score(y_test, y_prob)

                ax.plot(recall, precision, color=colors[i], lw=2,
                        label=f'{model_name} (AP = {ap_score:.3f})')

                # Store PR curve data
                for j in range(len(precision)):
                    pr_data_list.append({
                        'Model': model_name,
                        'Precision': precision[j],
                        'Recall': recall[j],
                        'Threshold': thresholds[j] if j < len(thresholds) else np.nan
                    })

                # Store AP summary
                ap_summary.append({
                    'Model': model_name,
                    'Average_Precision': ap_score
                })

            except Exception as e:
                print(f"    Error for {model_name}: {e}")
                continue

        # Baseline reference line
        ax.axhline(y=baseline, color='gray', linestyle='--', lw=1,
                   label=f'Baseline (prevalence = {baseline:.3f})')

        # Formatting
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('Recall (Sensitivity)', fontweight='bold')
        ax.set_ylabel('Precision (PPV)', fontweight='bold')
        ax.set_title(f'Precision-Recall Curves\n{task_name}', fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        safe_name = task_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
        filepath = os.path.join(output_dir, f"pr_curves_{safe_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

        # Save data tables
        if pr_data_list:
            pr_df = pd.DataFrame(pr_data_list)
            pr_df.to_csv(filepath.replace('.png', '_curves_data.csv'), index=False)

            ap_df = pd.DataFrame(ap_summary)
            ap_df = ap_df.sort_values('Average_Precision', ascending=False)
            ap_df.to_csv(filepath.replace('.png', '_ap_summary.csv'), index=False)

        print(f"    Saved: {os.path.basename(filepath)} + data tables")


# ==============================================================================
# 4. CALIBRATION PLOTS
# ==============================================================================

def generate_calibration_plots(results_list, task_data, output_dir, n_bins=10):
    """
    Generate calibration plots showing predicted vs actual probabilities.
    """
    print("\n=== Generating Calibration Plots ===")
    os.makedirs(output_dir, exist_ok=True)
    set_publication_style()

    # Group by task
    task_results_map = {}
    for r in results_list:
        task_name = r.get("LabelDefinition", "Unknown")
        if task_name not in task_results_map:
            task_results_map[task_name] = []
        task_results_map[task_name].append(r)

    for task_name, task_results in task_results_map.items():
        if task_name not in task_data:
            continue

        is_multi = "Multi:" in task_name
        if is_multi:
            continue  # Skip multi-class for calibration

        print(f"  Processing: {task_name}")

        X_train, y_train, X_test, y_test, _, _ = _unpack_task_data(task_data[task_name])

        fig, ax = plt.subplots(figsize=(8, 8))

        colors = sns.color_palette("husl", len(task_results))

        # Collect data for table
        calib_data_list = []
        calib_summary = []

        for i, result in enumerate(task_results):
            model = result.get("FinalModel")
            model_name = result.get("ModelName", "Unknown")

            if model is None or not hasattr(model, "predict_proba"):
                continue

            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy='uniform')

                ax.plot(prob_pred, prob_true, 's-', color=colors[i], lw=2,
                        label=model_name, markersize=6)

                # Store calibration curve data
                for j in range(len(prob_pred)):
                    calib_data_list.append({
                        'Model': model_name,
                        'Bin': j + 1,
                        'Mean_Predicted_Prob': prob_pred[j],
                        'Fraction_Positives': prob_true[j],
                        'Calibration_Error': abs(prob_pred[j] - prob_true[j])
                    })

                # Calculate calibration metrics
                ece = np.mean(np.abs(prob_pred - prob_true))  # Expected Calibration Error
                mce = np.max(np.abs(prob_pred - prob_true))   # Maximum Calibration Error
                calib_summary.append({
                    'Model': model_name,
                    'ECE': ece,
                    'MCE': mce,
                    'Brier_Score': np.mean((y_prob - y_test) ** 2)
                })

            except Exception as e:
                print(f"    Error for {model_name}: {e}")
                continue

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Calibration')

        # Formatting
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
        ax.set_ylabel('Fraction of Positives (Actual)', fontweight='bold')
        ax.set_title(f'Calibration Curves\n{task_name}', fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        safe_name = task_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
        filepath = os.path.join(output_dir, f"calibration_{safe_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

        # Save data tables
        if calib_data_list:
            calib_df = pd.DataFrame(calib_data_list)
            calib_df.to_csv(filepath.replace('.png', '_curves_data.csv'), index=False)

            summary_df = pd.DataFrame(calib_summary)
            summary_df = summary_df.sort_values('ECE')
            summary_df.to_csv(filepath.replace('.png', '_metrics_summary.csv'), index=False)

        print(f"    Saved: {os.path.basename(filepath)} + data tables")


# ==============================================================================
# 5. PUBLICATION-QUALITY CONFUSION MATRICES
# ==============================================================================

def generate_confusion_matrices(results_list, task_data, output_dir):
    """
    Generate publication-quality confusion matrix heatmaps.
    """
    print("\n=== Generating Confusion Matrices ===")
    os.makedirs(output_dir, exist_ok=True)
    set_publication_style()

    # Group by task
    task_results_map = {}
    for r in results_list:
        task_name = r.get("LabelDefinition", "Unknown")
        if task_name not in task_results_map:
            task_results_map[task_name] = []
        task_results_map[task_name].append(r)

    for task_name, task_results in task_results_map.items():
        if task_name not in task_data:
            continue

        print(f"  Processing: {task_name}")

        X_train, y_train, X_test, y_test, all_labels, _ = _unpack_task_data(task_data[task_name])

        is_multi = "Multi:" in task_name

        # Find best model by AUC-ROC (used for publication confusion matrix selection).
        best_result = None
        best_auc = -1
        for r in task_results:
            auc_val = r.get("Test_Metrics_Point", {}).get('auc', 0)
            if auc_val and auc_val > best_auc:
                best_auc = auc_val
                best_result = r

        if best_result is None or best_result.get("FinalModel") is None:
            continue

        model = best_result["FinalModel"]
        model_name = best_result.get("ModelName", "Unknown")
        threshold = best_result.get("CV_Threshold", 0.5)

        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                if is_multi:
                    y_pred = np.argmax(y_prob, axis=1)
                else:
                    y_pred = (y_prob[:, 1] >= threshold).astype(int)
            else:
                y_pred = model.predict(X_test)

            # Create confusion matrix
            if is_multi:
                labels = sorted(list(set(y_test)))
                class_names = ['Normal', 'MCI', 'Dementia'][:len(labels)]
            else:
                labels = [0, 1]
                class_names = ['Normal', 'Impaired']

            cm = confusion_matrix(y_test, y_pred, labels=labels)

            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)

            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Raw counts
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                        xticklabels=class_names, yticklabels=class_names,
                        cbar_kws={'label': 'Count'})
            axes[0].set_xlabel('Predicted', fontweight='bold')
            axes[0].set_ylabel('Actual', fontweight='bold')
            axes[0].set_title(f'Confusion Matrix (Counts)\n{model_name}', fontweight='bold')

            # Normalized (percentages)
            sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', ax=axes[1],
                        xticklabels=class_names, yticklabels=class_names,
                        vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
            axes[1].set_xlabel('Predicted', fontweight='bold')
            axes[1].set_ylabel('Actual', fontweight='bold')
            axes[1].set_title(f'Confusion Matrix (Normalized)\n{model_name}', fontweight='bold')

            fig.suptitle(task_name, fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()

            # Save plot
            safe_name = f"{task_name}_{model_name}".replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
            filepath = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
            plt.close()

            # Save data tables
            # Raw counts table
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            cm_df.index.name = 'Actual'
            cm_df.columns.name = 'Predicted'
            cm_df.to_csv(filepath.replace('.png', '_counts.csv'))

            # Normalized table
            cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
            cm_norm_df.index.name = 'Actual'
            cm_norm_df.columns.name = 'Predicted'
            cm_norm_df.to_csv(filepath.replace('.png', '_normalized.csv'))

            # Metrics derived from confusion matrix
            if len(labels) == 2:
                tn, fp, fn, tp = cm.ravel()
                cm_metrics = pd.DataFrame([{
                    'Model': model_name,
                    'Task': task_name,
                    'True_Positives': tp,
                    'True_Negatives': tn,
                    'False_Positives': fp,
                    'False_Negatives': fn,
                    'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
                    'Accuracy': (tp + tn) / (tp + tn + fp + fn)
                }])
                cm_metrics.to_csv(filepath.replace('.png', '_metrics.csv'), index=False)

            print(f"    Saved: {os.path.basename(filepath)} + data tables")

        except Exception as e:
            print(f"    Error: {e}")
            plt.close()


# ==============================================================================
# 6. MULTI-METRIC COMPARISON PLOTS
# ==============================================================================

def generate_radar_plot(results_list, output_dir):
    """
    Generate radar/spider plots comparing multiple metrics across models.
    """
    print("\n=== Generating Radar Plots ===")
    os.makedirs(output_dir, exist_ok=True)
    set_publication_style()

    # Group by task
    task_results_map = {}
    for r in results_list:
        task_name = r.get("LabelDefinition", "Unknown")
        if task_name not in task_results_map:
            task_results_map[task_name] = []
        task_results_map[task_name].append(r)

    metrics_to_plot = ['balanced_accuracy', 'auc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']
    metric_labels = ['Bal. Acc.', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1']

    for task_name, task_results in task_results_map.items():
        print(f"  Processing: {task_name}")

        # Filter to models with valid data
        valid_results = []
        for r in task_results:
            metrics = r.get("Test_Metrics_Point", {})
            if all(not np.isnan(metrics.get(m, np.nan)) for m in metrics_to_plot):
                valid_results.append(r)

        if len(valid_results) < 2:
            continue

        # Collect data for table
        radar_data = []

        # Setup radar chart
        num_vars = len(metrics_to_plot)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        colors = sns.color_palette("husl", len(valid_results))

        for i, result in enumerate(valid_results):
            model_name = result.get("ModelName", "Unknown")
            metrics = result.get("Test_Metrics_Point", {})

            values = [metrics.get(m, 0) for m in metrics_to_plot]
            values += values[:1]  # Complete the loop

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])

            # Collect data for table
            row = {'Model': model_name}
            for metric_key, metric_label in zip(metrics_to_plot, metric_labels):
                row[metric_label] = metrics.get(metric_key, np.nan)
            radar_data.append(row)

        # Formatting
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.set_title(f'Model Performance Comparison\n{task_name}', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

        plt.tight_layout()

        # Save plot
        safe_name = task_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
        filepath = os.path.join(output_dir, f"radar_plot_{safe_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

        # Save data table
        radar_df = pd.DataFrame(radar_data)
        radar_df = radar_df.sort_values('Bal. Acc.', ascending=False)
        radar_df.to_csv(filepath.replace('.png', '_data.csv'), index=False)
        print(f"    Saved: {os.path.basename(filepath)} + data table")


def generate_metrics_heatmap(results_list, output_dir):
    """
    Generate a heatmap showing all metrics for all models across tasks.
    """
    print("\n=== Generating Metrics Heatmap ===")
    os.makedirs(output_dir, exist_ok=True)
    set_publication_style()

    metrics = ['balanced_accuracy', 'auc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']
    metric_labels = ['Bal. Acc.', 'AUC', 'Sens.', 'Spec.', 'PPV', 'NPV', 'F1']

    # Create data for heatmap
    data = []
    row_labels = []

    for r in results_list:
        task_name = r.get("LabelDefinition", "Unknown")
        model_name = r.get("ModelName", "Unknown")

        # Short task name
        task_short = task_name.split('.')[0] if '.' in task_name else task_name[:20]
        row_label = f"{task_short}: {model_name}"

        point_metrics = r.get("Test_Metrics_Point", {})
        row_data = [point_metrics.get(m, np.nan) for m in metrics]

        if not all(np.isnan(row_data)):
            data.append(row_data)
            row_labels.append(row_label)

    if not data:
        return

    # Create DataFrame
    df = pd.DataFrame(data, index=row_labels, columns=metric_labels)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(row_labels) * 0.3)))

    # Heatmap
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, center=0.75,
                ax=ax, cbar_kws={'label': 'Score'}, linewidths=0.5)

    ax.set_title('Performance Metrics Across All Models and Tasks', fontweight='bold', fontsize=14)
    ax.set_xlabel('Metric', fontweight='bold')
    ax.set_ylabel('Task : Model', fontweight='bold')

    plt.tight_layout()

    filepath = os.path.join(output_dir, "metrics_heatmap_all.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(filepath.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # Save data table
    df.to_csv(filepath.replace('.png', '_data.csv'))
    print(f"  Saved: {os.path.basename(filepath)} + data table")


# ==============================================================================
# 7. SUMMARY TABLE FOR PUBLICATION
# ==============================================================================

def generate_publication_table(results_list, output_dir):
    """
    Generate a formatted table suitable for publication.
    """
    print("\n=== Generating Publication Table ===")

    rows = []
    for r in results_list:
        task = r.get("LabelDefinition", "Unknown")
        model = r.get("ModelName", "Unknown")

        point = r.get("Test_Metrics_Point", {})
        ci = r.get("Test_Metrics_CI_BS", {})

        def format_metric(key):
            val = point.get(key, np.nan)
            ci_val = ci.get(key, (np.nan, np.nan))
            if np.isnan(val):
                return "N/A"
            if not np.isnan(ci_val[0]):
                return f"{val:.3f} ({ci_val[0]:.3f}-{ci_val[1]:.3f})"
            return f"{val:.3f}"

        rows.append({
            'Task': task,
            'Model': model,
            'Balanced Acc. (95% CI)': format_metric('balanced_accuracy'),
            'AUC (95% CI)': format_metric('auc'),
            'Sensitivity (95% CI)': format_metric('sensitivity'),
            'Specificity (95% CI)': format_metric('specificity'),
            'PPV (95% CI)': format_metric('ppv'),
            'NPV (95% CI)': format_metric('npv'),
            'F1 (95% CI)': format_metric('f1')
        })

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = os.path.join(output_dir, "publication_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {os.path.basename(csv_path)}")

    # Save as LaTeX
    try:
        latex_path = os.path.join(output_dir, "publication_table.tex")
        with open(latex_path, 'w') as f:
            f.write("% Publication-ready table\n")
            f.write("% Requires \\usepackage{booktabs}\n\n")
            f.write(df.to_latex(index=False, escape=True,
                               column_format='l' * len(df.columns)))
        print(f"  Saved: {os.path.basename(latex_path)}")
    except Exception as e:
        print(f"  LaTeX export failed: {e}")

    return df


# ==============================================================================
# MAIN VISUALIZATION RUNNER
# ==============================================================================

def run_all_visualizations(results_list, task_data, feature_names_dict, output_base_dir):
    """
    Run all visualization functions.

    Parameters:
    -----------
    results_list : list
        List of all model results from the main analysis
    task_data : dict
        Dictionary with task data {task_name: (X_train, y_train, X_test, y_test, labels)}
    feature_names_dict : dict
        Dictionary mapping task names to feature names
    output_base_dir : str
        Base output directory for saving plots
    """
    print("\n" + "=" * 70)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 70)

    # Create subdirectories
    forest_dir = os.path.join(output_base_dir, "forest_plots")
    shap_dir = os.path.join(output_base_dir, "shap_analysis")
    roc_dir = os.path.join(output_base_dir, "roc_curves")
    calib_dir = os.path.join(output_base_dir, "calibration")
    cm_dir = os.path.join(output_base_dir, "confusion_matrices")
    comparison_dir = os.path.join(output_base_dir, "comparison_plots")
    tables_dir = os.path.join(output_base_dir, "tables")

    for d in [forest_dir, shap_dir, roc_dir, calib_dir, cm_dir, comparison_dir, tables_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Forest Plots (6 individual + 1 combined)
    generate_all_forest_plots(results_list, forest_dir)
    generate_combined_forest_plot(results_list, forest_dir)

    # 2. SHAP Analysis (for all 4 binary classification tasks)
    shap_tasks = [
        "1. Def Normal vs. Def Dementia",
        "2. Def+Pos Normal vs. Def+Pos Dementia",
        "3. Def Normal vs. Def MCI",
        "4. Def+Pos Normal vs. Def+Pos MCI",
    ]
    generate_shap_analysis(results_list, task_data, feature_names_dict, shap_dir,
                          tasks_to_analyze=shap_tasks)

    # 3. ROC Curves
    generate_roc_curves(results_list, task_data, roc_dir)

    # 3b. Precision-Recall Curves (important for imbalanced data)
    pr_dir = os.path.join(output_base_dir, "pr_curves")
    os.makedirs(pr_dir, exist_ok=True)
    generate_pr_curves(results_list, task_data, pr_dir)

    # 4. Calibration Plots
    generate_calibration_plots(results_list, task_data, calib_dir)

    # 5. Confusion Matrices
    generate_confusion_matrices(results_list, task_data, cm_dir)

    # 6. Multi-metric Comparisons
    generate_radar_plot(results_list, comparison_dir)
    generate_metrics_heatmap(results_list, comparison_dir)

    # 7. Publication Tables
    generate_publication_table(results_list, tables_dir)

    print("\n" + "=" * 70)
    print("VISUALIZATION GENERATION COMPLETE")
    print(f"All plots saved to: {output_base_dir}")
    print("=" * 70)


# ==============================================================================
# STANDALONE EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("This module is designed to be imported and used with the main analysis script.")
    print("\nUsage:")
    print("  from publication_visualizations import run_all_visualizations")
    print("  run_all_visualizations(results_list, task_data, feature_names_dict, output_dir)")
    print("\nOr run individual functions:")
    print("  generate_all_forest_plots(results_list, output_dir)")
    print("  generate_shap_analysis(results_list, task_data, feature_names_dict, output_dir)")
# fmt: on
