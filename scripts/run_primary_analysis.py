# fmt: off
"""
PARALLELIZED VERSION - Optimized for High-Performance Server (32 CPUs / 100GB RAM)
Key changes:
1. Parallel model training loop using joblib (8 models at a time)
2. Parallelized bootstrap iterations (8 workers per model)
3. Inner threading set to 2 for BLAS/LAPACK operations in tree-based models
4. GridSearchCV uses 2 parallel CV folds
"""

import os
import sys

# Add project root to Python path so scripts can import from utils/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Force unbuffered output (see progress in real-time)
os.environ["PYTHONUNBUFFERED"] = "1"

# ============================================================================
# HIGH-PERFORMANCE SERVER OPTIMIZATION (32 CPUs / 100GB RAM)
# ============================================================================
# 8 parallel model evaluations × 2 threads each = 16 CPUs for training.
# Bootstrap and GridSearchCV share remaining CPUs.
# Total peak usage: ~24 CPUs, ~24 GB RAM (well within 100 GB).
N_PARALLEL_MODELS = 8

# Allow 2 threads per model for BLAS/LAPACK operations (matrix ops in
# tree-based models, SVM kernel computations, logistic regression solvers).
N_THREADS_PER_MODEL = 2

# Set environment variables for multi-threaded BLAS
os.environ["OMP_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS_PER_MODEL)

# Match XGBoost/LightGBM threading to per-model budget
os.environ["XGB_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["LGBM_NUM_THREADS"] = str(N_THREADS_PER_MODEL)

import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for parallel processing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     RepeatedStratifiedKFold, GridSearchCV)
from sklearn.metrics import (roc_auc_score, confusion_matrix, precision_recall_curve,
                             auc, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, average_precision_score)
from scipy import stats  # For BCa bootstrap CI calculations
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier, HistGradientBoostingClassifier,
                              VotingClassifier)

try:
    from xgboost import XGBClassifier
except ImportError:
    print("NOTE: XGBoost not installed. XGBoost models will not be available.")
    XGBClassifier = None

# LightGBM - often better than XGBoost for imbalanced multiclass
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    LGBMClassifier = None
    print("NOTE: LightGBM not installed. LightGBM models will not be available.")
    print("      Install with: pip install lightgbm")

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
import collections
import time

# Parallelization imports
from joblib import Parallel, delayed, dump as joblib_dump

# SMOTE for imbalanced data (optional)
# IMPORTANT: We use imblearn.pipeline.Pipeline to ensure SMOTE is applied
# INSIDE cross-validation folds (only to training data), not before CV.
# This prevents data leakage from synthetic samples.
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    SMOTE = None
    ImbPipeline = None
    print("NOTE: imbalanced-learn not installed. SMOTE oversampling not available.")
    print("      Install with: pip install imbalanced-learn")

# Import publication visualizations module
try:
    from utils.publication_visualizations import run_all_visualizations, SHAP_AVAILABLE
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False
    SHAP_AVAILABLE = False
    print("NOTE: publication_visualizations module not found. Advanced visualizations will be skipped.")
    print("      Place publication_visualizations.py in the same directory to enable.")

# Import statistical tests module
try:
    from utils.statistical_tests import (
        run_statistical_comparisons,
        extract_all_feature_importances
    )
    STATISTICAL_TESTS_AVAILABLE = True
except ImportError:
    STATISTICAL_TESTS_AVAILABLE = False
    print("NOTE: statistical_tests module not found. Statistical comparisons will be skipped.")
    print("      Place statistical_tests.py in the same directory to enable.")

# Import epidemiology utilities module
try:
    from utils.epidemiology_utils import run_epidemiology_analysis
    EPIDEMIOLOGY_UTILS_AVAILABLE = True
except ImportError:
    EPIDEMIOLOGY_UTILS_AVAILABLE = False
    print("NOTE: epidemiology_utils module not found. Epidemiological analysis will be skipped.")
    print("      Place epidemiology_utils.py in the same directory to enable.")

# Import thesis tables and figures module
try:
    from utils.thesis_tables_figures import run_thesis_outputs
    THESIS_OUTPUTS_AVAILABLE = True
except ImportError:
    THESIS_OUTPUTS_AVAILABLE = False
    run_thesis_outputs = None  # type: ignore
    print("NOTE: thesis_tables_figures module not found. Thesis outputs will be skipped.")
    print("      Place thesis_tables_figures.py in the same directory to enable.")

warnings.filterwarnings('ignore')

# --- Configuration ---
OPTIMIZE_THRESHOLD_METRIC = 'balanced_accuracy'
GRIDSEARCH_SCORING_METRIC = 'balanced_accuracy'

# Imbalance detection threshold: if minority class < this proportion, flag as imbalanced
IMBALANCE_THRESHOLD = 0.3  # e.g., if minority class is < 30% of majority

# When True, PR-AUC is highlighted as primary metric for imbalanced data
# ROC-AUC can be misleadingly high with class imbalance; PR-AUC is more informative
PREFER_PR_AUC_FOR_IMBALANCED = True

N_BOOTSTRAPS = 500  # Reduced for faster runtime on ARC (was 1000)
N_CV_SPLITS_GRIDSEARCH = 3
N_CV_SPLITS_THRESH = 5
N_CV_REPEATS_THRESH = 3
TEST_SET_SIZE = 0.3
RANDOM_STATE = 42
MCNEMAR_ALPHA = 0.05

# Parallelization settings for 32-CPU server
# Strategy: Parallelize at all levels — enough CPUs to avoid contention
N_JOBS_GRIDSEARCH = 2  # 2 parallel CV folds within GridSearchCV
N_JOBS_BOOTSTRAP = 8   # 8 parallel bootstrap workers per model
N_JOBS_MODELS = N_PARALLEL_MODELS  # 8 parallel model evaluations

print(f"--- Configuration: High-Performance Server (32 CPUs / 100GB RAM) ---")
print(f"Parallel Models: {N_PARALLEL_MODELS} | Threads/Model: {N_THREADS_PER_MODEL} | Bootstrap Workers: {N_JOBS_BOOTSTRAP} | GridSearch Jobs: {N_JOBS_GRIDSEARCH}")

# ---------------------------------------
# 1. Core Helper Functions
# ---------------------------------------

def find_optimal_threshold(y_true, y_prob, metric=OPTIMIZE_THRESHOLD_METRIC):
    """Find the optimal probability threshold for binary classification.

    Sweeps thresholds from 0.01 to 0.99 and returns the one that maximizes
    the chosen metric (default: balanced accuracy).

    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    metric : str
        Metric to optimize. One of 'f1', 'balanced_accuracy', 'youden'.

    Returns:
    --------
    float
        Optimal threshold value (0.5 if optimization fails).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(np.unique(y_true)) < 2:
        return 0.5

    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = 0
        try:
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_true, y_pred)
            elif metric == 'youden':
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    score = sensitivity + specificity - 1
                else: score = 0
            else:
                raise ValueError(f"Unknown metric: {metric}")
        except Exception:
            score = 0
        scores.append(score)

    if not scores: return 0.5
    scores = np.array(scores)
    if np.all(np.isnan(scores)): return 0.5

    best_score = np.nanmax(scores)
    best_threshold_indices = np.where(np.isclose(scores, best_score, equal_nan=False))[0]
    best_thresholds = thresholds[best_threshold_indices]
    middle_best_idx = np.argmin(np.abs(best_thresholds - 0.5))
    optimal_threshold = best_thresholds[middle_best_idx]

    return optimal_threshold


def find_threshold_cv(model, X_train, y_train, metric=OPTIMIZE_THRESHOLD_METRIC,
                      n_splits=N_CV_SPLITS_THRESH, n_repeats=N_CV_REPEATS_THRESH, random_state=RANDOM_STATE):
    """Find optimal threshold using Repeated Stratified K-Fold CV on training data."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_thresholds = []
    is_dataframe = hasattr(X_train, 'iloc')

    for train_idx_inner, val_idx_inner in cv.split(X_train, y_train):
        if is_dataframe:
            X_train_inner, X_val_inner = X_train.iloc[train_idx_inner], X_train.iloc[val_idx_inner]
            y_train_inner, y_val_inner = (y_train.iloc[train_idx_inner], y_train.iloc[val_idx_inner]) if hasattr(y_train, 'iloc') \
                                         else (y_train[train_idx_inner], y_train[val_idx_inner])
        else:
            X_train_inner, X_val_inner = X_train[train_idx_inner], X_train[val_idx_inner]
            y_train_inner, y_val_inner = y_train[train_idx_inner], y_train[val_idx_inner]

        if len(np.unique(y_val_inner)) < 2:
            continue

        try:
            model_clone_thresh = clone(model)
            model_clone_thresh.fit(X_train_inner, y_train_inner)

            if hasattr(model_clone_thresh, "predict_proba"):
                y_proba_val = model_clone_thresh.predict_proba(X_val_inner)[:, 1]
                if np.all(np.isfinite(y_proba_val)):
                    optimal_thresh_fold = find_optimal_threshold(y_val_inner, y_proba_val, metric=metric)
                    fold_thresholds.append(optimal_thresh_fold)
        except Exception:
            continue

    if not fold_thresholds:
        return 0.5

    final_threshold = np.median(fold_thresholds)
    return final_threshold


def detect_class_imbalance(y, threshold=IMBALANCE_THRESHOLD):
    """
    Detect class imbalance and return recommendations for metric selection.

    Parameters:
    -----------
    y : array-like
        Target labels
    threshold : float
        If minority class proportion < threshold, consider imbalanced

    Returns:
    --------
    dict with:
        - is_imbalanced: bool
        - imbalance_ratio: float (majority/minority)
        - minority_proportion: float
        - class_counts: dict
        - recommended_primary_metric: str
        - metric_warning: str or None
    """
    y = np.asarray(y)
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))

    total = len(y)
    minority_count = min(counts)
    majority_count = max(counts)

    minority_proportion = minority_count / total
    imbalance_ratio = majority_count / minority_count if minority_count > 0 else np.inf

    is_imbalanced = minority_proportion < threshold

    # Recommendations based on imbalance
    if is_imbalanced:
        recommended_metric = 'pr_auc'
        metric_warning = (
            f"WARNING: Class imbalance detected (ratio {imbalance_ratio:.1f}:1, "
            f"minority={minority_proportion:.1%}). PR-AUC is recommended over ROC-AUC. "
            f"ROC-AUC may be misleadingly optimistic for imbalanced data."
        )
    else:
        recommended_metric = 'auc'
        metric_warning = None

    return {
        'is_imbalanced': is_imbalanced,
        'imbalance_ratio': imbalance_ratio,
        'minority_proportion': minority_proportion,
        'class_counts': class_counts,
        'recommended_primary_metric': recommended_metric,
        'metric_warning': metric_warning
    }


def calculate_metrics_binary(y_true, y_pred_prob, y_pred_label):
    """Calculate standard binary classification metrics."""
    metrics = {}
    y_true = np.asarray(y_true)
    unique_true = np.unique(y_true)

    if y_pred_prob is not None and len(unique_true) > 1:
        y_pred_prob = np.asarray(y_pred_prob)
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_prob)
        except ValueError:
            metrics['auc'] = np.nan
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
            metrics['pr_auc'] = auc(recall, precision)
        except ValueError:
            metrics['pr_auc'] = np.nan
    else:
        metrics['auc'] = np.nan
        metrics['pr_auc'] = np.nan

    if y_pred_label is not None:
        y_pred_label = np.asarray(y_pred_label)
        try:
            cm = confusion_matrix(y_true, y_pred_label, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            else:
                metrics['sensitivity'], metrics['specificity'], metrics['ppv'], metrics['npv'] = np.nan, np.nan, np.nan, np.nan
        except ValueError:
            metrics['sensitivity'], metrics['specificity'], metrics['ppv'], metrics['npv'] = np.nan, np.nan, np.nan, np.nan

        metrics['f1'] = f1_score(y_true, y_pred_label, zero_division=0)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred_label)
    else:
        metrics['sensitivity'], metrics['specificity'], metrics['ppv'], metrics['npv'] = np.nan, np.nan, np.nan, np.nan
        metrics['f1'] = np.nan
        metrics['balanced_accuracy'] = np.nan

    return metrics


def calculate_metrics_multi(y_true, y_pred_prob, y_pred_label, all_original_labels, verbose=False):
    """Calculate standard multi-class classification metrics (macro averages).

    If verbose=True, also prints per-class metrics for debugging.
    """
    metrics = {}
    y_true = np.asarray(y_true)
    unique_true = np.unique(y_true)
    if all_original_labels is None: all_original_labels = sorted(unique_true)

    # Class label names for debugging output
    class_names = {0: 'Normal', 1: 'MCI', 2: 'Dementia'}

    if y_pred_prob is not None and len(unique_true) > 1 and y_pred_prob.shape[1] >= len(all_original_labels):
        y_pred_prob = np.asarray(y_pred_prob)
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_prob, multi_class='ovr', average='weighted', labels=all_original_labels)
        except ValueError:
            metrics['auc'] = np.nan

        pr_auc_values = []
        test_labels_present = unique_true
        for cls_idx, cls_label in enumerate(all_original_labels):
            if cls_label in test_labels_present:
                y_true_cls = (y_true == cls_label).astype(int)
                if len(np.unique(y_true_cls)) < 2:
                    pr_auc_values.append(np.nan)
                    continue
                if cls_idx < y_pred_prob.shape[1]:
                    y_proba_cls = y_pred_prob[:, cls_idx]
                    try:
                        precision, recall, _ = precision_recall_curve(y_true_cls, y_proba_cls)
                        pr_auc_values.append(auc(recall, precision))
                    except ValueError:
                        pr_auc_values.append(np.nan)
                else:
                    pr_auc_values.append(np.nan)
            else:
                pr_auc_values.append(np.nan)

        valid_pr_aucs = [v for v in pr_auc_values if not np.isnan(v)]
        metrics['pr_auc'] = np.mean(valid_pr_aucs) if valid_pr_aucs else np.nan
    else:
        metrics['auc'] = np.nan
        metrics['pr_auc'] = np.nan

    if y_pred_label is not None:
        y_pred_label = np.asarray(y_pred_label)
        metrics['f1'] = f1_score(y_true, y_pred_label, average='macro', zero_division=0, labels=all_original_labels)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred_label)
        metrics['ppv'] = precision_score(y_true, y_pred_label, average='macro', zero_division=0, labels=all_original_labels)
        metrics['sensitivity'] = recall_score(y_true, y_pred_label, average='macro', zero_division=0, labels=all_original_labels)

        spec_vals, npv_vals = [], []
        test_labels_present = unique_true
        for cls_label in all_original_labels:
            if cls_label in test_labels_present:
                y_true_cls = (y_true == cls_label).astype(int)
                y_pred_cls = (y_pred_label == cls_label).astype(int)
                if len(np.unique(y_true_cls)) < 2:
                    spec_vals.append(np.nan)
                    npv_vals.append(np.nan)
                    continue
                try:
                    cm_cls = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
                    if cm_cls.shape == (2, 2):
                        tn, fp, fn, tp = cm_cls.ravel()
                        spec_vals.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                        npv_vals.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
                    else:
                        spec_vals.append(np.nan)
                        npv_vals.append(np.nan)
                except ValueError:
                    spec_vals.append(np.nan)
                    npv_vals.append(np.nan)
            else:
                spec_vals.append(np.nan)
                npv_vals.append(np.nan)

        valid_spec = [v for v in spec_vals if not np.isnan(v)]
        valid_npv = [v for v in npv_vals if not np.isnan(v)]
        metrics['specificity'] = np.mean(valid_spec) if valid_spec else np.nan
        metrics['npv'] = np.mean(valid_npv) if valid_npv else np.nan

        # Per-class metrics for debugging
        if verbose and y_pred_label is not None:
            print("    Per-class performance:")
            for cls_label in all_original_labels:
                cls_name = class_names.get(cls_label, f"Class_{cls_label}")
                y_true_cls = (y_true == cls_label).astype(int)
                y_pred_cls = (y_pred_label == cls_label).astype(int)
                n_true = np.sum(y_true_cls)
                n_pred = np.sum(y_pred_cls)
                correct = np.sum((y_true_cls == 1) & (y_pred_cls == 1))
                recall = correct / n_true if n_true > 0 else 0
                precision = correct / n_pred if n_pred > 0 else 0
                print(f"      {cls_name}: Recall={recall:.3f}, Precision={precision:.3f}, "
                      f"True={n_true}, Pred={n_pred}, Correct={correct}")
    else:
        metrics['sensitivity'], metrics['specificity'], metrics['ppv'], metrics['npv'] = np.nan, np.nan, np.nan, np.nan
        metrics['f1'] = np.nan
        metrics['balanced_accuracy'] = np.nan

    return metrics


def _single_bootstrap_iteration(i, final_model, X_test, y_test, fixed_threshold,
                                 seed, multi_class, all_original_labels, metric_names):
    """Single bootstrap iteration - designed for parallel execution."""
    try:
        rng = np.random.RandomState(seed)
        n_test = len(y_test)
        boot_indices = rng.choice(n_test, n_test, replace=True)

        if hasattr(X_test, 'iloc'):
            X_boot_test = X_test.iloc[boot_indices]
            y_boot_test = y_test.iloc[boot_indices] if hasattr(y_test, 'iloc') else y_test[boot_indices]
        else:
            X_boot_test = X_test[boot_indices]
            y_boot_test = y_test[boot_indices]

        y_pred_prob_boot = None
        y_pred_label_boot = None

        if hasattr(final_model, "predict_proba"):
            y_pred_prob_boot = final_model.predict_proba(X_boot_test)

        if multi_class:
            if y_pred_prob_boot is not None:
                y_pred_label_boot = np.argmax(y_pred_prob_boot, axis=1)
            elif hasattr(final_model, "predict"):
                y_pred_label_boot = final_model.predict(X_boot_test)
            else:
                return {key: np.nan for key in metric_names}

            metrics_iter = calculate_metrics_multi(y_boot_test, y_pred_prob_boot, y_pred_label_boot, all_original_labels)

        else:
            if y_pred_prob_boot is not None:
                if y_pred_prob_boot.shape[1] < 2:
                    return {key: np.nan for key in metric_names}
                y_prob_pos_boot = y_pred_prob_boot[:, 1]
                current_threshold_bs = fixed_threshold if fixed_threshold is not None and not np.isnan(fixed_threshold) else 0.5
                y_pred_label_boot = (y_prob_pos_boot >= current_threshold_bs).astype(int)
                metrics_iter = calculate_metrics_binary(y_boot_test, y_prob_pos_boot, y_pred_label_boot)
            elif hasattr(final_model, "predict"):
                y_pred_label_boot = final_model.predict(X_boot_test)
                metrics_iter = calculate_metrics_binary(y_boot_test, None, y_pred_label_boot)
            else:
                return {key: np.nan for key in metric_names}

        return metrics_iter

    except Exception:
        return {key: np.nan for key in metric_names}


def bootstrap_test_metrics_parallel(final_model, X_test, y_test, fixed_threshold,
                                    n_bootstraps=N_BOOTSTRAPS, random_state=RANDOM_STATE,
                                    multi_class=False, all_original_labels=None, n_jobs=N_JOBS_BOOTSTRAP):
    """
    PARALLELIZED: Perform bootstrapping on the TEST SET to estimate CIs for metrics.
    Uses joblib to parallelize bootstrap iterations.
    """
    metric_names = ['auc', 'pr_auc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'balanced_accuracy']
    n_test = len(y_test)

    if n_test == 0:
        return {key: np.full(n_bootstraps, np.nan) for key in metric_names}

    # Generate reproducible seeds for each bootstrap
    base_rng = np.random.RandomState(random_state)
    seeds = base_rng.randint(0, 2**31 - 1, size=n_bootstraps)

    # Run bootstrap iterations in parallel
    results_list = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_single_bootstrap_iteration)(
            i, final_model, X_test, y_test, fixed_threshold,
            seeds[i], multi_class, all_original_labels, metric_names
        )
        for i in range(n_bootstraps)
    )

    # Aggregate results
    bootstrapped_results = {key: [] for key in metric_names}
    for metrics_iter in results_list:
        for key in metric_names:
            bootstrapped_results[key].append(metrics_iter.get(key, np.nan))

    for key in metric_names:
        bootstrapped_results[key] = np.array(bootstrapped_results[key])

    return bootstrapped_results


def median_iqr(values):
    """Calculate median and IQR, ignoring NaNs."""
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0: return np.nan, np.nan
    med = np.median(valid_values)
    q75, q25 = np.percentile(valid_values, [75, 25])
    return med, q75 - q25


def compute_ci(values, alpha=0.05, method='bca'):
    """
    Compute confidence intervals, ignoring NaNs.

    Parameters:
    -----------
    values : array
        Bootstrap sample values
    alpha : float
        Significance level (default 0.05 for 95% CI)
    method : str
        'percentile' for simple percentile CI (faster, assumes symmetry)
        'bca' for Bias-Corrected and Accelerated CI (better for bounded/skewed metrics)

    Returns:
    --------
    tuple: (lower, upper) CI bounds
    """
    valid_values = values[~np.isnan(values)]
    if len(valid_values) < 2:
        return np.nan, np.nan

    if method == 'percentile':
        # Simple percentile method (original)
        lower = np.percentile(valid_values, 100 * alpha/2)
        upper = np.percentile(valid_values, 100 * (1 - alpha/2))
        return lower, upper

    elif method == 'bca':
        # Bias-Corrected and Accelerated (BCa) bootstrap CI
        # Better for bounded statistics like AUC, sensitivity, specificity
        try:
            n_boot = len(valid_values)
            original_stat = np.median(valid_values)  # Use median as point estimate

            # Bias correction factor (z0)
            prop_less = np.mean(valid_values < original_stat)
            prop_less = np.clip(prop_less, 0.001, 0.999)
            z0 = stats.norm.ppf(prop_less)

            # Acceleration factor (a) - simplified estimation using skewness
            # Full jackknife is expensive; use skewness approximation (Efron 1987)
            skewness = stats.skew(valid_values)
            a = skewness / 6.0
            a = np.clip(a, -0.4, 0.4)  # Bound to prevent extreme adjustments

            # BCa adjusted percentiles
            z_alpha_lower = stats.norm.ppf(alpha / 2)
            z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

            def adjusted_percentile(z_alpha):
                numerator = z0 + z_alpha
                denominator = 1 - a * (z0 + z_alpha)
                if abs(denominator) < 1e-10:
                    return 0.5
                return stats.norm.cdf(z0 + numerator / denominator)

            p_lower = adjusted_percentile(z_alpha_lower)
            p_upper = adjusted_percentile(z_alpha_upper)

            # Clip to valid range
            p_lower = np.clip(p_lower, 0.001, 0.999)
            p_upper = np.clip(p_upper, 0.001, 0.999)

            lower = np.percentile(valid_values, 100 * p_lower)
            upper = np.percentile(valid_values, 100 * p_upper)

            return lower, upper

        except Exception:
            # Fall back to percentile method if BCa fails
            lower = np.percentile(valid_values, 100 * alpha/2)
            upper = np.percentile(valid_values, 100 * (1 - alpha/2))
            return lower, upper

    else:
        raise ValueError(f"Unknown CI method: {method}. Use 'percentile' or 'bca'.")


# ---------------------------------------
# Jaakkimainen Benchmark Implementation
# ---------------------------------------

class JaakkimainenBenchmark(BaseEstimator, ClassifierMixin):
    """
    Rule-based benchmark classifier implementing the Jaakkimainen algorithm.

    The Jaakkimainen algorithm predicts dementia if ANY of the following indicators
    are present (value > 0):
    - jaak_hosp_code: Hospital diagnosis code for dementia
    - jaak_rx_filled: Dementia-related prescription filled
    - jaak_3claims+_2yr_30days: 3+ claims within 2 years with 30-day gap

    This is a deterministic rule-based classifier (no training required).
    """

    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        """No fitting required for rule-based classifier."""
        return self

    def predict(self, X):
        """
        Predict dementia (1) if any of the 3 Jaakkimainen columns is > 0.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 3)
            The 3-column Jaakkimainen indicator array:
            [jaak_hosp_code, jaak_rx_filled, jaak_3claims+_2yr_30days]

        Returns:
        --------
        y_pred : array of shape (n_samples,)
            1 if any indicator > 0 (Dementia), else 0 (Normal)
        """
        X = np.asarray(X)
        # Predict 1 (Dementia) if ANY column is > 0
        y_pred = (np.any(X > 0, axis=1)).astype(int)
        return y_pred

    def predict_proba(self, X):
        """
        Return deterministic probabilities based on the rule.

        Parameters:
        -----------
        X : array-like of shape (n_samples, 3)
            The 3-column Jaakkimainen indicator array

        Returns:
        --------
        proba : array of shape (n_samples, 2)
            Probabilities [P(Normal), P(Dementia)] - deterministic 0.0 or 1.0
        """
        y_pred = self.predict(X)
        proba = np.zeros((len(y_pred), 2))
        proba[:, 1] = y_pred.astype(float)  # P(Dementia)
        proba[:, 0] = 1.0 - proba[:, 1]      # P(Normal)
        return proba


def evaluate_benchmark_rule(Z_test, y_test, label_definition, n_bootstraps=N_BOOTSTRAPS,
                            random_state=RANDOM_STATE):
    """
    Evaluate the Jaakkimainen benchmark rule on the test set with bootstrapped CIs.

    This function mirrors the ML model evaluation by:
    1. Calculating point metrics on the full test set
    2. Running the same bootstrap procedure used for ML models
    3. Computing BCa confidence intervals

    Parameters:
    -----------
    Z_test : array-like of shape (n_samples, 3)
        Jaakkimainen indicator columns for test set
    y_test : array-like of shape (n_samples,)
        True labels for test set (0=Normal, 1=Dementia)
    label_definition : str
        Name of the classification task
    n_bootstraps : int
        Number of bootstrap iterations (default: N_BOOTSTRAPS)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    dict : Result dictionary matching ML model result format
    """
    start_time = time.time()

    # Initialize the benchmark model
    benchmark_model = JaakkimainenBenchmark()

    # Get predictions on full test set
    y_pred = benchmark_model.predict(Z_test)
    y_pred_proba = benchmark_model.predict_proba(Z_test)
    y_prob_pos = y_pred_proba[:, 1]

    # Calculate point metrics
    point_metrics = calculate_metrics_binary(y_test, y_prob_pos, y_pred)

    # Run bootstrap using the same parallel function as ML models.
    # NOTE: Although the Jaakkimainen algorithm is deterministic (same input
    # always gives the same prediction), bootstrapping is still meaningful here.
    # Each bootstrap iteration resamples the TEST SET with replacement, creating
    # different test compositions. The resulting CIs capture sampling uncertainty
    # in the performance metrics — i.e., how stable the measured sensitivity,
    # specificity, etc. would be across different random samples of patients from
    # the same population. This is standard practice for evaluating any fixed
    # classifier (rule-based or trained) and ensures comparable CIs with the ML
    # models, which are also bootstrapped on the same test set.
    bs_test_metrics = bootstrap_test_metrics_parallel(
        final_model=benchmark_model,
        X_test=Z_test,  # Pass Z_test as X_test - model.predict expects 3-column array
        y_test=y_test,
        fixed_threshold=0.5,  # Rule-based, threshold doesn't apply but needed for function
        n_bootstraps=n_bootstraps,
        random_state=random_state,
        multi_class=False,
        all_original_labels=None,
        n_jobs=N_JOBS_BOOTSTRAP
    )

    # Compute CIs and median/IQR from bootstrap results
    metrics_ci = {}
    metrics_median_iqr = {}
    metric_names = list(bs_test_metrics.keys())

    for name in metric_names:
        values = np.asarray(bs_test_metrics.get(name, []))
        if values.size > 0:
            valid_values = values[~np.isnan(values)]
            if valid_values.size > 1:
                lower_ci, upper_ci = compute_ci(valid_values)
                median, iqr = median_iqr(valid_values)
                metrics_ci[name] = (lower_ci, upper_ci)
                metrics_median_iqr[name] = (median, iqr)
            elif valid_values.size == 1:
                median, iqr = median_iqr(valid_values)
                metrics_ci[name] = (np.nan, np.nan)
                metrics_median_iqr[name] = (median, iqr)
            else:
                metrics_ci[name] = (np.nan, np.nan)
                metrics_median_iqr[name] = (np.nan, np.nan)
        else:
            metrics_ci[name] = (np.nan, np.nan)
            metrics_median_iqr[name] = (np.nan, np.nan)

    elapsed = time.time() - start_time

    # Build result dictionary matching ML model format
    result = {
        "LabelDefinition": label_definition,
        "ModelName": "Jaakkimainen Benchmark",
        "MultiClass": False,
        "CV_Threshold": np.nan,  # Not applicable for rule-based
        "BestParams": "Rule-based (no tuning)",
        "FinalModel": benchmark_model,
        "TrainSupport": {},  # Not applicable
        "TestSupport": dict(zip(*np.unique(y_test, return_counts=True))),
        "Test_Metrics_Point": point_metrics,
        "Bootstrap_Test_Metrics": bs_test_metrics,
        "Test_Metrics_Median_IQR_BS": metrics_median_iqr,
        "Test_Metrics_CI_BS": metrics_ci,
        "RunTimeSec": elapsed,
        "Error": ""
    }

    return result


def create_smote_pipeline(base_pipeline, random_state=RANDOM_STATE):
    """
    Wrap a sklearn Pipeline with SMOTE using imblearn's Pipeline.

    IMPORTANT: This ensures SMOTE is applied INSIDE cross-validation folds,
    only to training data. This prevents data leakage from synthetic samples
    appearing in validation folds.

    The imblearn Pipeline:
    - Applies SMOTE only during fit() (training)
    - Skips SMOTE during predict()/transform() (inference)
    - Works correctly with GridSearchCV cross-validation

    Returns the original pipeline if SMOTE is not available.
    """
    if not SMOTE_AVAILABLE or ImbPipeline is None or SMOTE is None:
        return base_pipeline

    try:
        # Extract steps from the base sklearn Pipeline
        if hasattr(base_pipeline, 'steps'):
            base_steps = list(base_pipeline.steps)
        else:
            # Not a pipeline, wrap the estimator directly
            base_steps = [('estimator', base_pipeline)]

        # Create SMOTE step with adaptive k_neighbors
        # k_neighbors will be set dynamically, but we use a safe default
        # The actual k is handled by SMOTE internally with fallback
        smote_step = ('smote', SMOTE(random_state=random_state, k_neighbors=5))

        # Create imblearn Pipeline: SMOTE -> base pipeline steps
        # SMOTE is applied first, then the rest of the pipeline
        smote_pipeline = ImbPipeline([smote_step] + base_steps)

        print(f"    SMOTE integrated into pipeline (applied inside CV folds only)")
        return smote_pipeline

    except Exception as e:
        print(f"    Warning: Could not create SMOTE pipeline: {e}")
        return base_pipeline


def create_adaptive_smote(y_train, random_state=RANDOM_STATE):
    """
    Create a SMOTE instance with k_neighbors adapted to the minority class size.
    Returns None if SMOTE cannot be applied (too few samples).
    """
    if not SMOTE_AVAILABLE or SMOTE is None:
        return None

    try:
        _, counts = np.unique(y_train, return_counts=True)
        min_count = np.min(counts)

        # SMOTE requires at least k_neighbors + 1 samples in minority class
        k_neighbors = min(5, min_count - 1)

        if k_neighbors < 1:
            print(f"    SMOTE skipped: minority class has only {min_count} samples")
            return None

        return SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    except Exception:
        return None


# ---------------------------------------
# 2. Main Evaluation Function (PARALLELIZED BOOTSTRAP)
# ---------------------------------------

def build_and_evaluate_model_cv(X_train, y_train, X_test, y_test,
                                model_pipeline, param_grid,
                                model_name, label_definition, feature_names,
                                multi_class=False, random_state=RANDOM_STATE,
                                all_original_labels=None):
    """
    Builds, tunes (GridSearchCV), and evaluates a model.
    Bootstrap is now parallelized.
    """
    start_time_model = time.time()
    results = {
        "LabelDefinition": label_definition, "ModelName": model_name,
        "MultiClass": multi_class, "CV_Threshold": np.nan,
        "BestParams": None, "FinalModel": None,
        "TrainSupport": {}, "TestSupport": {},
        "Test_Metrics_Point": {}, "Bootstrap_Test_Metrics": {},
        "Test_Metrics_Median_IQR_BS": {}, "Test_Metrics_CI_BS": {},
        "RunTimeSec": 0, "Error": ""
    }

    if X_train is None or y_train is None or X_test is None or y_test is None or X_train.shape[0]==0 or X_test.shape[0]==0:
        error_msg = "Missing or empty train/test data."
        results["Error"] = error_msg
        results["RunTimeSec"] = time.time() - start_time_model
        return results

    try:
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        results["TrainSupport"] = dict(zip(map(str, unique_train), counts_train))
        results["TestSupport"] = dict(zip(map(str, unique_test), counts_test))

        min_classes_needed = len(all_original_labels) if multi_class and all_original_labels else 2
        if len(unique_train) < min_classes_needed or len(unique_test) < min_classes_needed:
            error_msg = f"Insufficient class variation in train ({len(unique_train)}) or test ({len(unique_test)}) set."
            results["Error"] = error_msg
            results["RunTimeSec"] = time.time() - start_time_model
            return results
    except Exception as e:
        error_msg = f"Data variation check failed: {e}"
        results["Error"] = error_msg
        results["RunTimeSec"] = time.time() - start_time_model
        return results

    if multi_class and all_original_labels is None:
        all_original_labels = sorted(np.unique(np.concatenate((y_train, y_test))))

    # --- 1. Hyperparameter Tuning using GridSearchCV ---
    best_model = None

    # NOTE: SMOTE is now applied INSIDE the pipeline (via imblearn.pipeline.Pipeline)
    # This ensures SMOTE only applies to training folds during CV, preventing data leakage.
    # Sample weights are still used for models that don't support class_weight natively.
    X_train_processed = X_train
    y_train_processed = y_train
    sample_weights = None
    fit_params = {}

    if multi_class:
        # Compute sample weights for class imbalance (for GBM and XGBoost which lack class_weight)
        # NOTE: Sample weights are ONLY used for models NOT wrapped with SMOTE.
        # Models wrapped with SMOTE should NOT use sample weights (SMOTE changes data size).
        sample_weights = compute_sample_weight('balanced', y_train_processed)

        # Helper function to find estimator step in pipeline (handles ImbPipeline too)
        def find_estimator_step(pipeline, estimator_keywords):
            """Find estimator step name in pipeline or ImbPipeline."""
            steps = None
            if hasattr(pipeline, 'steps'):
                steps = pipeline.steps
            elif hasattr(pipeline, 'named_steps'):
                steps = list(pipeline.named_steps.items())

            if steps:
                for step_name, step in steps:
                    step_type = str(type(step)).lower()
                    for keyword in estimator_keywords:
                        if keyword in step_type:
                            return step_name
            return None

        # Check if model needs sample_weight (GBM, XGBoost) AND is NOT wrapped with SMOTE
        # SMOTE pipelines use ImbPipeline which has 'smote' as first step
        is_smote_pipeline = False
        if hasattr(model_pipeline, 'steps'):
            first_step_name = model_pipeline.steps[0][0] if model_pipeline.steps else ''
            is_smote_pipeline = 'smote' in first_step_name.lower()

        if not is_smote_pipeline:
            # Only apply sample weights to non-SMOTE pipelines
            gbm_step = find_estimator_step(model_pipeline, ['gradientboosting'])
            xgb_step = find_estimator_step(model_pipeline, ['xgb'])

            if gbm_step and 'hist' not in gbm_step.lower():
                fit_params[f'{gbm_step}__sample_weight'] = sample_weights
            elif xgb_step:
                fit_params[f'{xgb_step}__sample_weight'] = sample_weights

    if not param_grid:
        try:
            best_model = clone(model_pipeline)
            if fit_params and sample_weights is not None:
                best_model.fit(X_train_processed, y_train_processed, **fit_params)
            else:
                best_model.fit(X_train_processed, y_train_processed)
            results["BestParams"] = "Default"
        except Exception as e:
            error_msg = f"Fitting default model failed: {e}"
            results["Error"] = error_msg
            results["RunTimeSec"] = time.time() - start_time_model
            return results
    else:
        try:
            cv_grid = StratifiedKFold(n_splits=N_CV_SPLITS_GRIDSEARCH, shuffle=True, random_state=random_state)
            gs_scoring = GRIDSEARCH_SCORING_METRIC
            if multi_class and gs_scoring == 'roc_auc':
                gs_scoring = 'roc_auc_ovr_weighted'
            elif multi_class and gs_scoring == 'f1':
                gs_scoring = 'f1_macro'

            grid_search = GridSearchCV(
                estimator=clone(model_pipeline),
                param_grid=param_grid,
                scoring=gs_scoring,
                cv=cv_grid,
                n_jobs=N_JOBS_GRIDSEARCH,  # LIMITED parallelism within GridSearch
                refit=True
            )

            # Pass sample weights for GBM if available
            if fit_params:
                grid_search.fit(X_train_processed, y_train_processed, **fit_params)
            else:
                grid_search.fit(X_train_processed, y_train_processed)
            best_model = grid_search.best_estimator_
            results["BestParams"] = grid_search.best_params_

        except Exception as e:
            error_msg = f"GridSearchCV failed: {e}"
            results["Error"] = error_msg
            results["RunTimeSec"] = time.time() - start_time_model
            return results

    results["FinalModel"] = best_model

    # --- 2. Find Threshold using CV on Training Data (if binary) ---
    cv_optimal_threshold = 0.5
    if not multi_class:
        try:
            cv_optimal_threshold = find_threshold_cv(
                best_model, X_train_processed, y_train_processed, metric=OPTIMIZE_THRESHOLD_METRIC,
                n_splits=N_CV_SPLITS_THRESH, n_repeats=N_CV_REPEATS_THRESH,
                random_state=random_state
            )
            results["CV_Threshold"] = cv_optimal_threshold
        except Exception as e:
            results["CV_Threshold"] = 0.5
            results["Error"] += f"; CV threshold finding failed: {e}"

    # --- 3. Evaluate Final Model on Test Set ---
    y_pred_prob_test = None
    y_pred_label_test = None
    test_metrics_point = {}

    try:
        if hasattr(best_model, "predict_proba"):
            y_pred_prob_test = best_model.predict_proba(X_test)

        if multi_class:
            if y_pred_prob_test is not None:
                y_pred_label_test = np.argmax(y_pred_prob_test, axis=1)
            elif hasattr(best_model, "predict"):
                y_pred_label_test = best_model.predict(X_test)
            else: raise ValueError("Model cannot predict")
            # Use verbose=True to see per-class performance for multiclass
            test_metrics_point = calculate_metrics_multi(y_test, y_pred_prob_test, y_pred_label_test, all_original_labels, verbose=True)
        else:
            if y_pred_prob_test is not None:
                if y_pred_prob_test.shape[1] >= 2:
                    y_prob_pos_test = y_pred_prob_test[:, 1]
                    current_threshold = results["CV_Threshold"] if not pd.isna(results["CV_Threshold"]) else 0.5
                    y_pred_label_test = (y_prob_pos_test >= current_threshold).astype(int)
                    test_metrics_point = calculate_metrics_binary(y_test, y_prob_pos_test, y_pred_label_test)
            elif hasattr(best_model, "predict"):
                y_pred_label_test = best_model.predict(X_test)
                test_metrics_point = calculate_metrics_binary(y_test, None, y_pred_label_test)

        results["Test_Metrics_Point"] = test_metrics_point

    except Exception as e:
        results["Error"] += f"; Test set evaluation failed: {e}"

    # --- 4. Bootstrap Test Set Results (PARALLELIZED) ---
    try:
        if results.get("FinalModel") is not None and y_test is not None and len(y_test) > 0:
            bs_test_metrics = bootstrap_test_metrics_parallel(
                best_model, X_test, y_test, results["CV_Threshold"],
                n_bootstraps=N_BOOTSTRAPS, random_state=random_state,
                multi_class=multi_class, all_original_labels=all_original_labels,
                n_jobs=N_JOBS_BOOTSTRAP
            )
            results["Bootstrap_Test_Metrics"] = bs_test_metrics

            metrics_ci = {}
            metrics_median_iqr = {}
            metric_names_bs = list(bs_test_metrics.keys())
            for name in metric_names_bs:
                values = bs_test_metrics.get(name, np.array([]))
                if values.size > 0:
                    valid_values = values[~np.isnan(values)]
                    if valid_values.size > 1:
                        lower_ci, upper_ci = compute_ci(valid_values)
                        median, iqr = median_iqr(valid_values)
                        metrics_ci[name] = (lower_ci, upper_ci)
                        metrics_median_iqr[name] = (median, iqr)
                    elif valid_values.size == 1:
                        median, iqr = median_iqr(valid_values)
                        metrics_ci[name] = (np.nan, np.nan)
                        metrics_median_iqr[name] = (median, iqr)
                    else:
                        metrics_ci[name] = (np.nan, np.nan)
                        metrics_median_iqr[name] = (np.nan, np.nan)
                else:
                    metrics_ci[name] = (np.nan, np.nan)
                    metrics_median_iqr[name] = (np.nan, np.nan)

            results["Test_Metrics_Median_IQR_BS"] = metrics_median_iqr
            results["Test_Metrics_CI_BS"] = metrics_ci

    except Exception as e:
        results["Error"] += f"; Bootstrap failed: {e}"

    end_time_model = time.time()
    results["RunTimeSec"] = end_time_model - start_time_model

    return results


# ---------------------------------------
# 3. Parallel Model Evaluation Wrapper
# ---------------------------------------

def evaluate_single_model_task(job_args):
    """
    Wrapper function for parallel model evaluation.
    Each call trains and evaluates one model on one task.
    """
    (label_def, model_name, model_instance, param_grid,
     X_train, y_train, X_test, y_test,
     current_feature_names, is_multi, all_orig_labels) = job_args

    print(f"  [PARALLEL] Starting: {model_name} on '{label_def}'")
    start = time.time()

    try:
        # Convert labels to int
        y_train_int = y_train.astype(int)
        y_test_int = y_test.astype(int)
        min_label = np.min(np.concatenate((y_train_int, y_test_int))) if len(y_train_int) > 0 else 0
        if min_label != 0:
            y_train_int = y_train_int - min_label
            y_test_int = y_test_int - min_label
            if all_orig_labels is not None:
                all_orig_labels = sorted([max(0, l - min_label) for l in all_orig_labels])

        model_to_run = clone(model_instance)

        res = build_and_evaluate_model_cv(
            X_train, y_train_int, X_test, y_test_int,
            model_pipeline=model_to_run,
            param_grid=param_grid,
            model_name=model_name,
            label_definition=label_def,
            feature_names=current_feature_names,
            multi_class=is_multi,
            random_state=RANDOM_STATE,
            all_original_labels=all_orig_labels
        )

        elapsed = time.time() - start
        print(f"  [PARALLEL] Finished: {model_name} on '{label_def}' in {elapsed:.1f}s")
        return res

    except Exception as e:
        elapsed = time.time() - start
        print(f"  [PARALLEL] FAILED: {model_name} on '{label_def}' after {elapsed:.1f}s - {e}")
        return {
            "LabelDefinition": label_def,
            "ModelName": model_name,
            "Error": str(e),
            "MultiClass": is_multi,
            "BestParams": "N/A",
            "Test_Metrics_Point": {},
            "Bootstrap_Test_Metrics": {},
            "Test_Metrics_Median_IQR_BS": {},
            "Test_Metrics_CI_BS": {},
            "RunTimeSec": elapsed
        }


# ---------------------------------------
# 4. Visualization Functions (unchanged from original - abbreviated)
# ---------------------------------------

def generate_auc_charts(results_list, output_dir):
    """Generate AUC distribution charts."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    for idx, result in enumerate(results_list):
        try:
            label_def = result.get("LabelDefinition", f"Task_{idx}")
            model_name = result.get("ModelName", f"Model_{idx}")
            safe_name = f"{label_def.replace('/', '_').replace(' ', '_').replace(':','_').replace('.','')}_{model_name.replace(' ', '_')}"

            bs_metrics = result.get("Bootstrap_Test_Metrics", {})
            bs_aucs = bs_metrics.get("auc", np.array([]))
            med_auc_bs = result.get("Test_Metrics_Median_IQR_BS", {}).get('auc', (np.nan, np.nan))[0]
            lower_ci_bs, upper_ci_bs = result.get("Test_Metrics_CI_BS", {}).get('auc', (np.nan, np.nan))

            valid_bs_aucs = bs_aucs[~np.isnan(bs_aucs)] if isinstance(bs_aucs, np.ndarray) else []

            if len(valid_bs_aucs) > 1:
                plt.figure(figsize=(10, 6))
                n_bins = min(30, max(10, len(valid_bs_aucs)//10 + 1)) if len(valid_bs_aucs) > 20 else 10
                sns.histplot(valid_bs_aucs, kde=True, stat="density", bins=n_bins)
                if not np.isnan(med_auc_bs): plt.axvline(med_auc_bs, color='r', linestyle='--', label=f'Median AUC: {med_auc_bs:.3f}')
                if not np.isnan(lower_ci_bs): plt.axvline(lower_ci_bs, color='g', linestyle=':', label=f'95% CI Lower: {lower_ci_bs:.3f}')
                if not np.isnan(upper_ci_bs): plt.axvline(upper_ci_bs, color='g', linestyle=':', label=f'95% CI Upper: {upper_ci_bs:.3f}')

                plt.title(f'Test Set Bootstrap AUC Distribution - {model_name}\n{label_def}', fontsize=14, wrap=True)
                plt.xlabel('AUC', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                if not np.all(np.isnan([med_auc_bs, lower_ci_bs, upper_ci_bs])):
                    plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{safe_name}_bootstrap_test_auc_dist.png"), dpi=300)
                plt.close()

        except Exception as e:
            print(f"Error generating AUC chart for model {idx}: {str(e)}")
            plt.close()


def generate_comparison_charts(results_list, output_dir):
    """Generate model comparison charts."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    tasks = {}
    for r in results_list:
        task_name = r["LabelDefinition"]
        if task_name not in tasks: tasks[task_name] = []
        tasks[task_name].append(r)

    for task_name, task_results in tasks.items():
        try:
            if not task_results: continue
            safe_task_name = task_name.replace('/', '_').replace(' ', '_').replace(':','_').replace('.','')
            model_names = [r["ModelName"] for r in task_results]

            median_aucs_bs = np.array([r.get("Test_Metrics_Median_IQR_BS", {}).get('auc', (np.nan, np.nan))[0] for r in task_results])
            lower_cis_bs = np.array([r.get("Test_Metrics_CI_BS", {}).get('auc', (np.nan, np.nan))[0] for r in task_results])
            upper_cis_bs = np.array([r.get("Test_Metrics_CI_BS", {}).get('auc', (np.nan, np.nan))[1] for r in task_results])

            valid_indices_ci = ~np.isnan(median_aucs_bs) & ~np.isnan(lower_cis_bs) & ~np.isnan(upper_cis_bs)
            if not np.any(valid_indices_ci):
                continue

            model_names_plot = [model_names[i] for i, valid in enumerate(valid_indices_ci) if valid]
            median_aucs_plot = median_aucs_bs[valid_indices_ci]
            lower_cis_plot = lower_cis_bs[valid_indices_ci]
            upper_cis_plot = upper_cis_bs[valid_indices_ci]

            if model_names_plot:
                plt.figure(figsize=(max(10, 1.5 * len(model_names_plot)), 7))
                x_pos = np.arange(len(model_names_plot))
                ci_half_width = (upper_cis_plot - lower_cis_plot) / 2.0
                ci_half_width = np.nan_to_num(ci_half_width, nan=0.0)
                ci_half_width = np.maximum(0, ci_half_width)

                bars = plt.bar(x_pos, median_aucs_plot, yerr=ci_half_width, align='center', alpha=0.8, capsize=5, color='cornflowerblue', edgecolor='black')
                plt.title(f'Model Comparison - Median Test Set AUC (Bootstrap)\n{task_name}', fontsize=14, wrap=True)
                plt.ylabel('Median Test Set AUC (Error Bars = 95% CI / 2)', fontsize=12)
                plt.xticks(x_pos, model_names_plot, rotation=45, fontsize=11, ha='right')
                min_auc_display = max(0.3, np.nanmin(lower_cis_plot) - 0.05) if len(lower_cis_plot) > 0 else 0.3
                plt.ylim(min_auc_display, 1.05)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{safe_task_name}_model_comparison_test_bar.png"), dpi=300)
                plt.close()

        except Exception as e:
            print(f"Error generating comparison charts for '{task_name}': {str(e)}")
            plt.close()


def create_comprehensive_metrics_table(results_list, output_dir):
    """Create a comprehensive metrics table."""
    rows = []
    metric_names = ['AUC', 'PR AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1 Score', 'Balanced Acc.']
    metric_keys_lower = ['auc', 'pr_auc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'balanced_accuracy']

    for result in results_list:
        row_data = collections.OrderedDict()
        row_data["Task"] = result.get("LabelDefinition", "Unknown Task")
        row_data["Model"] = result.get("ModelName", "Unknown Model")

        point_metrics = result.get("Test_Metrics_Point", {})
        bs_medians = result.get("Test_Metrics_Median_IQR_BS", {})
        bs_cis = result.get("Test_Metrics_CI_BS", {})

        for name, key_lower in zip(metric_names, metric_keys_lower):
            point_val = point_metrics.get(key_lower, np.nan)
            bs_med_tuple = bs_medians.get(key_lower, (np.nan, np.nan))
            bs_ci_tuple = bs_cis.get(key_lower, (np.nan, np.nan))
            bs_med = bs_med_tuple[0]
            bs_low, bs_up = bs_ci_tuple

            point_str = f"{point_val:.3f}" if not pd.isna(point_val) else "N/A"
            if not pd.isna(bs_med) and not pd.isna(bs_low) and not pd.isna(bs_up):
                bs_str = f"[{bs_med:.3f} ({bs_low:.3f}-{bs_up:.3f})]"
            elif not pd.isna(point_val):
                bs_str = "[BS N/A]"
            else:
                bs_str = "[N/A]"

            row_data[f"{name} (Point [BS Median (CI)])"] = f"{point_str} {bs_str}"

        train_support_str = ", ".join([f"{k}: {v}" for k, v in sorted(result.get("TrainSupport", {}).items(), key=lambda item: str(item[0]))]) or "N/A"
        test_support_str = ", ".join([f"{k}: {v}" for k, v in sorted(result.get("TestSupport", {}).items(), key=lambda item: str(item[0]))]) or "N/A"
        row_data["Train Support"] = train_support_str
        row_data["Test Support"] = test_support_str

        cv_thresh = result.get('CV_Threshold')
        is_multi = result.get("MultiClass", False)
        thresh_str = f"{cv_thresh:.3f}" if not is_multi and cv_thresh is not None and not pd.isna(cv_thresh) else "N/A"
        row_data[f"Threshold ({OPTIMIZE_THRESHOLD_METRIC})"] = thresh_str

        best_params = result.get("BestParams")
        if best_params == "Default":
            params_str = "Default"
        elif isinstance(best_params, dict):
            params_str = "; ".join([f"{k.split('__')[-1]}={v}" for k, v in sorted(best_params.items())])
        elif best_params:
            params_str = str(best_params)
        else:
            params_str = "N/A"
        row_data["Best Params"] = params_str

        row_data["Multi-Class"] = "Yes" if is_multi else "No"
        row_data["Run Time (s)"] = f"{result.get('RunTimeSec', 0):.1f}"
        row_data["Error"] = result.get("Error", "")

        rows.append(row_data)

    if not rows:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows)
    metrics_csv_path = os.path.join(output_dir, f"final_evaluation_metrics_{OPTIMIZE_THRESHOLD_METRIC}_thresh_tuned_PARALLEL.csv")
    try:
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Comprehensive evaluation metrics saved to: {metrics_csv_path}")
    except Exception as e:
        print(f"\nError saving final metrics CSV: {e}")

    return metrics_df


# ---------------------------------------
# 5. Data Preparation Functions (same as original)
# ---------------------------------------

def prepare_data_common(subset_df, label_definition, exclude_cols,
                        jaak_cols=None):
    """Handles common steps: dropping exclusions, extracting Jaakkimainen cols.

    NOTE: Imputation and one-hot encoding are NOT done here to avoid data
    leakage. They are performed AFTER train/test split via
    preprocess_after_split().

    Parameters:
    -----------
    subset_df : pd.DataFrame
        The subset of data for this task
    label_definition : str
        Name/description of the classification task
    exclude_cols : list
        Columns to exclude from features
    jaak_cols : list, optional
        Jaakkimainen benchmark column names to extract before dropping.
        If provided, returns Z_df containing these columns aligned with X.

    Returns:
    --------
    If jaak_cols is None: (X_df, y_series, None)
    If jaak_cols provided: (X_df, y_series, None, Z_df)

    Note: The third return value (feature_names) is None here and will be
    populated after preprocessing in preprocess_after_split().
    """
    if subset_df is None or subset_df.empty:
        if jaak_cols:
            return None, None, None, None
        return None, None, None

    # Extract Jaakkimainen columns BEFORE dropping exclusions
    Z_df = None
    if jaak_cols:
        jaak_cols_present = [col for col in jaak_cols if col in subset_df.columns]
        if jaak_cols_present:
            Z_df = subset_df[jaak_cols_present].copy()
            # Fill missing values with 0 and convert to int
            Z_df = Z_df.fillna(0).astype(int)
            # If some columns are missing, add them as zeros
            for col in jaak_cols:
                if col not in Z_df.columns:
                    Z_df[col] = 0
            # Ensure column order matches jaak_cols
            Z_df = Z_df[jaak_cols]
        else:
            # No Jaakkimainen columns found - create zero DataFrame
            Z_df = pd.DataFrame(0, index=subset_df.index, columns=jaak_cols)

    cols_to_drop_existing = [col for col in exclude_cols if col in subset_df.columns]
    X_df = subset_df.drop(columns=cols_to_drop_existing, errors='ignore')
    y_series = subset_df["cognitive_status"]

    y_series = y_series.loc[X_df.index]

    # Align Z_df index with X_df if jaak_cols provided
    if jaak_cols and Z_df is not None:
        Z_df = Z_df.loc[X_df.index]
        return X_df, y_series, None, Z_df

    return X_df, y_series, None


def preprocess_after_split(X_train_df, X_test_df, label_definition):
    """Impute missing values and one-hot encode AFTER train/test split.

    Fits imputation medians and encoding categories on training data only,
    then transforms both train and test to prevent data leakage.

    Parameters:
    -----------
    X_train_df : pd.DataFrame
        Training features (raw, with possible NaNs and categorical columns)
    X_test_df : pd.DataFrame
        Test features (raw, with possible NaNs and categorical columns)
    label_definition : str
        Name/description of the classification task (for error messages)

    Returns:
    --------
    (X_train_encoded, X_test_encoded, feature_names_encoded) or (None, None, None)
    """
    if X_train_df is None or X_test_df is None:
        return None, None, None

    X_train = X_train_df.copy()
    X_test = X_test_df.copy()

    # --- Numeric imputation (fit medians on TRAINING data only) ---
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        train_medians = {}
        for col in numeric_cols:
            median_val = X_train[col].median()
            if pd.isna(median_val):
                median_val = 0
            train_medians[col] = median_val
            # Impute train
            if X_train[col].isnull().any():
                X_train[col] = X_train[col].fillna(median_val)
            # Impute test using TRAINING medians
            if X_test[col].isnull().any():
                X_test[col] = X_test[col].fillna(median_val)

    # --- Categorical imputation (same logic, applied independently) ---
    categorical_cols_train = X_train.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols_train.empty:
        for col in categorical_cols_train:
            X_train[col] = X_train[col].fillna('missing').astype(str)
            X_train[col] = X_train[col].replace(['nan', 'NaN', 'None', '', 'None'], 'missing')
    categorical_cols_test = X_test.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols_test.empty:
        for col in categorical_cols_test:
            X_test[col] = X_test[col].fillna('missing').astype(str)
            X_test[col] = X_test[col].replace(['nan', 'NaN', 'None', '', 'None'], 'missing')

    # --- One-hot encode (fit on TRAINING categories, align test to match) ---
    try:
        cols_to_encode_train = X_train.select_dtypes(include=['object', 'category']).columns
        cols_to_encode_test = X_test.select_dtypes(include=['object', 'category']).columns

        if not cols_to_encode_train.empty or not cols_to_encode_test.empty:
            # Encode training data
            X_train_encoded = pd.get_dummies(X_train, columns=cols_to_encode_train,
                                              drop_first=True, dummy_na=False, dtype=int)
            # Encode test data
            X_test_encoded = pd.get_dummies(X_test, columns=cols_to_encode_test,
                                             drop_first=True, dummy_na=False, dtype=int)

            # Align test columns to match training columns exactly:
            # - Add any columns present in train but missing in test (set to 0)
            # - Drop any columns present in test but not in train (unseen categories)
            train_cols = X_train_encoded.columns
            for col in train_cols:
                if col not in X_test_encoded.columns:
                    X_test_encoded[col] = 0
            X_test_encoded = X_test_encoded[train_cols]
        else:
            X_train_encoded = X_train
            X_test_encoded = X_test

        feature_names_encoded = list(X_train_encoded.columns)

    except Exception as e:
        print(f"    ERROR during encoding for '{label_definition}': {e}")
        return None, None, None

    return X_train_encoded, X_test_encoded, feature_names_encoded


def finalize_binary_data(X_encoded, y_original_series, label_positive_list):
    """Convert string cognitive_status labels to binary 0/1.

    Parameters:
    -----------
    X_encoded : pd.DataFrame
        Feature matrix (passed through unchanged).
    y_original_series : pd.Series
        Raw cognitive_status string labels.
    label_positive_list : list of str
        Status values to map to class 1 (e.g., ["definite_dementia"]).

    Returns:
    --------
    (X_encoded, y_binary) or (None, None) if fewer than 2 classes remain.
    """
    if X_encoded is None or y_original_series is None: return None, None
    label_positive_list_clean = [s.lower().strip().replace(' ', '_') for s in label_positive_list]
    y_binary = y_original_series.apply(lambda x: 1 if x in label_positive_list_clean else 0).values
    unique_final, counts_final = np.unique(y_binary, return_counts=True)
    if len(unique_final) < 2:
        return None, None
    return X_encoded, y_binary


def finalize_multiclass_data(X_encoded, y_original_series):
    """Convert string cognitive_status labels to integer classes.

    Maps: 'normal' -> 0, 'mci' -> 1, 'dementia' -> 2.
    Rows that don't match any category are removed.

    Parameters:
    -----------
    X_encoded : pd.DataFrame
        Feature matrix (rows with unmapped labels are dropped).
    y_original_series : pd.Series
        Raw cognitive_status string labels.

    Returns:
    --------
    (X_final, y_final, sorted_labels) or (None, None, None) if no valid rows.
    """
    if X_encoded is None or y_original_series is None: return None, None, None

    def map_to_three_class(status):
        if "dementia" in status: return 2
        elif "mci" in status: return 1
        elif "normal" in status: return 0
        else: return -1

    y_multi = y_original_series.apply(map_to_three_class).values
    valid_indices = (y_multi != -1)

    if np.sum(~valid_indices) > 0:
        X_encoded_filtered = X_encoded[valid_indices]
        y_multi_filtered = y_multi[valid_indices]
        if X_encoded_filtered.empty:
            return None, None, None
        X_final = X_encoded_filtered
        y_final = y_multi_filtered
    else:
        X_final = X_encoded
        y_final = y_multi

    all_orig_numeric_labels = sorted(list(set(y_final)))
    unique_final = np.unique(y_final)
    if len(unique_final) < 2:
        return None, None, None

    return X_final, y_final, all_orig_numeric_labels


# ---------------------------------------
# 6. Main Execution Function (PARALLELIZED)
# ---------------------------------------

def main():
    """
    PARALLELIZED main execution function.
    Uses joblib to run model evaluations in parallel.
    """
    print("=" * 60)
    print("PARALLELIZED ANALYSIS PIPELINE")
    print(f"Using {N_JOBS_MODELS} parallel model evaluations")
    print(f"Using {N_JOBS_BOOTSTRAP} parallel bootstrap workers per model")
    print(f"Using {N_JOBS_GRIDSEARCH} jobs per GridSearchCV")
    print("=" * 60)

    analysis_start_time = time.time()

    # --- Setup Paths ---
    data_dir = os.path.join(PROJECT_ROOT, "data")
    possible_data_paths = [
        os.path.join(data_dir, "MAIN_new.csv"),
        os.path.join(data_dir, "MAIN.csv"),
    ]
    data_path = None
    for p in possible_data_paths:
        if os.path.exists(p):
            data_path = p
            print(f"Found data file at: {os.path.abspath(data_path)}")
            break

    if data_path is None:
        print(f"CRITICAL ERROR: CSV data file not found in {data_dir}")
        print("Place MAIN_new.csv or MAIN.csv in the data/ directory.")
        return

    # --- Output Directories ---
    base_output_dir = os.path.join(PROJECT_ROOT, "output", "primary")
    charts_dir = os.path.join(base_output_dir, "charts_final_eval")
    results_dir = os.path.join(base_output_dir, "results_final_eval")

    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Output will be saved in: {base_output_dir}")

    # --- Load Data ---
    print("\n=== Loading Data ===")
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        df['cognitive_status'] = df['cognitive_status'].fillna('unknown').astype(str)
        df['cognitive_status'] = df['cognitive_status'].str.lower().str.strip().str.replace(' ', '_', regex=False)
        print("Value counts for 'cognitive_status':")
        print(df['cognitive_status'].value_counts())
    except Exception as e:
        print(f"CRITICAL ERROR loading data: {e}")
        return

    # --- Define Exclusions ---
    exclude_columns_for_X = sorted(list(set([
        # Identifiers and dates
        "id", "prompt_id", "prompt_visitdate", "visit_year", "jaak_hosp_code", "dob",
        # Label/outcome columns (leakage)
        "dementia", "cognitive_label", "dem/norm/oth", "jaak_Dem", "cognitive_status",
        "jaak_rx_filled", "jaak_3claims+_2yr_30days",
        # Demographics (excluded per study design)
        "race", "education_years", "education_level", "sex", "age",
        # Cognitive test scores (label leakage - used to determine cognitive_status)
        "moca_total", "mmsetotal",
        # Other non-predictive variables
        "living_arrangement",
    ])))

    # --- Prepare Task Subsets ---
    print("\n=== Preparing Task Subsets ===")
    task_dfs = collections.OrderedDict()
    task_dfs["1. Def Normal vs. Def Dementia"] = df[df["cognitive_status"].isin(["definite_normal", "definite_dementia"])].copy()
    task_dfs["2. Def+Pos Normal vs. Def+Pos Dementia"] = df[df["cognitive_status"].isin(["definite_normal", "possible_normal", "definite_dementia", "possible_dementia"])].copy()
    task_dfs["3. Def Normal vs. Def MCI"] = df[df["cognitive_status"].isin(["definite_normal", "definite_mci"])].copy()
    task_dfs["4. Def+Pos Normal vs. Def+Pos MCI"] = df[df["cognitive_status"].isin(["definite_normal", "possible_normal", "definite_mci", "possible_mci"])].copy()
    task_dfs["5. Multi: Def Normal vs. Def MCI vs. Def Dementia"] = df[df["cognitive_status"].isin(["definite_normal", "definite_mci", "definite_dementia"])].copy()
    task_dfs["6. Multi: Def+Pos Normal vs. Def+Pos MCI vs. Def+Pos Dementia"] = df[df["cognitive_status"].isin(["definite_normal", "possible_normal", "definite_mci", "possible_mci", "definite_dementia", "possible_dementia"])].copy()

    # --- Process and Split Data ---
    print("\n=== Preprocessing and Splitting Data ===")
    task_data = {}
    feature_names_dict_final = {}

    # Jaakkimainen benchmark columns - extract for binary Dementia vs Normal tasks
    JAAK_COLS = ['jaak_hosp_code', 'jaak_rx_filled', 'jaak_3claims+_2yr_30days']

    for label_def, df_subset in task_dfs.items():
        print(f"\n  Processing: {label_def}")
        if df_subset.empty:
            continue

        # Check if this is a binary Dementia vs Normal task (for Jaakkimainen benchmark)
        is_dementia_vs_normal = ("Dementia" in label_def and "Normal" in label_def
                                  and "Multi:" not in label_def and "MCI" not in label_def)

        # Step 1: Extract features (X_df), labels (y_ser), and Jaakkimainen cols (Z_df)
        # NOTE: No imputation or encoding here — that happens AFTER the split
        if is_dementia_vs_normal:
            result = prepare_data_common(df_subset, label_def, exclude_columns_for_X,
                                         jaak_cols=JAAK_COLS)
            if result[0] is None:
                continue
            X_raw_df, y_ser, _, Z_df = result
        else:
            result = prepare_data_common(df_subset, label_def, exclude_columns_for_X)
            if result[0] is None:
                continue
            X_raw_df, y_ser, _ = result
            Z_df = None

        is_multi = "Multi:" in label_def
        all_orig_numeric_labels_task = None

        # Step 2: Finalize labels (string -> numeric)
        if is_multi:
            X_final_df, y_final_np, all_orig_numeric_labels_task = finalize_multiclass_data(X_raw_df, y_ser)
            Z_final_df = None
        else:
            pos_labels_str = []
            if "Dementia" in label_def and "Normal" in label_def:
                pos_labels_str = ["definite_dementia", "possible_dementia"]
            elif "MCI" in label_def and "Normal" in label_def:
                pos_labels_str = ["definite_mci", "possible_mci"]
            if not pos_labels_str:
                continue
            X_final_df, y_final_np = finalize_binary_data(X_raw_df, y_ser, pos_labels_str)

            # Align Z_df with finalized data if available
            if Z_df is not None and X_final_df is not None:
                Z_final_df = Z_df.loc[X_final_df.index]
            else:
                Z_final_df = None

        if X_final_df is None or y_final_np is None:
            continue

        try:
            # Step 3: Split RAW DataFrames (before imputation/encoding)
            if Z_final_df is not None:
                X_train_raw_df, X_test_raw_df, y_train, y_test, Z_train_df, Z_test_df = train_test_split(
                    X_final_df, y_final_np, Z_final_df,
                    test_size=TEST_SET_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=y_final_np
                )
                Z_test = Z_test_df.values
            else:
                X_train_raw_df, X_test_raw_df, y_train, y_test = train_test_split(
                    X_final_df, y_final_np,
                    test_size=TEST_SET_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=y_final_np
                )
                Z_test = None

            # Step 4: Impute and encode AFTER split (fit on train only)
            X_train_enc_df, X_test_enc_df, feat_names = preprocess_after_split(
                X_train_raw_df, X_test_raw_df, label_def
            )
            if X_train_enc_df is None:
                print(f"    ERROR: Preprocessing after split failed for '{label_def}'")
                continue

            feature_names_dict_final[label_def] = feat_names

            # Convert to numpy arrays for downstream compatibility
            X_train = X_train_enc_df.values
            X_test = X_test_enc_df.values

            # Store task data (add Z_test as 6th element for applicable tasks)
            task_data[label_def] = (X_train, y_train, X_test, y_test, all_orig_numeric_labels_task, Z_test)
            print(f"    Split complete: Train={X_train.shape[0]}, Test={X_test.shape[0]}, Features={X_train.shape[1]}")
            if Z_test is not None:
                print(f"    Jaakkimainen benchmark data extracted: Z_test shape={Z_test.shape}")
        except Exception as e:
            print(f"    ERROR splitting/preprocessing: {e}")
            continue

    if not task_data:
        print("CRITICAL ERROR: No task datasets prepared. Exiting.")
        return

    print(f"\nSuccessfully prepared {len(task_data)} tasks.")

    # --- Define Models ---
    print("\n=== Defining Models ===")
    max_iter_lr = 3000

    lr_pipeline = Pipeline([('scaler', StandardScaler()), ('logisticregression', LogisticRegression(solver='liblinear', max_iter=max_iter_lr, random_state=RANDOM_STATE, class_weight='balanced', penalty='l1'))])
    rf_pipeline = Pipeline([('randomforestclassifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=N_THREADS_PER_MODEL))])
    gbm_pipeline = Pipeline([('gradientboostingclassifier', GradientBoostingClassifier(random_state=RANDOM_STATE))])
    svm_pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'))])

    if XGBClassifier:
        xgb_pipeline = Pipeline([('scaler', StandardScaler()), ('xgbclassifier', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=N_THREADS_PER_MODEL))])
    else:
        xgb_pipeline = None

    # HistGradientBoosting for binary - has native class_weight support
    hist_gbm_binary_pipeline = Pipeline([
        ('histgradientboostingclassifier', HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            max_iter=200,
            max_depth=6,
            min_samples_leaf=10,
            learning_rate=0.05,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15
        ))
    ])

    # LightGBM for binary - has native class_weight support
    if LGBM_AVAILABLE and LGBMClassifier:
        lgbm_binary_pipeline = Pipeline([
            ('lgbmclassifier', LGBMClassifier(
                objective='binary',
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                n_jobs=N_THREADS_PER_MODEL,
                verbose=-1
            ))
        ])
    else:
        lgbm_binary_pipeline = None

    # Random Forest with entropy criterion for binary
    rf_entropy_binary_pipeline = Pipeline([
        ('randomforestclassifier', RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight='balanced_subsample',
            criterion='entropy',
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=3,
            min_samples_split=5,
            max_features='sqrt',
            n_jobs=N_THREADS_PER_MODEL,
            oob_score=True
        ))
    ])

    base_estimators_stack = [
        ('lr_stack', clone(lr_pipeline)),
        ('rf_stack', clone(rf_pipeline)),
        ('gbm_stack', clone(gbm_pipeline)),
        ('svm_stack', clone(svm_pipeline))
    ]
    if xgb_pipeline:
        base_estimators_stack.append(('xgb_stack', clone(xgb_pipeline)))

    meta_learner_lr_pipeline = Pipeline([('scaler_meta', StandardScaler()), ('logisticregression_meta', LogisticRegression(solver='liblinear', max_iter=max_iter_lr, random_state=RANDOM_STATE, class_weight='balanced'))])
    stacking_pipeline = StackingClassifier(estimators=base_estimators_stack, final_estimator=meta_learner_lr_pipeline, passthrough=False, cv=StratifiedKFold(n_splits=N_CV_SPLITS_GRIDSEARCH, shuffle=True, random_state=RANDOM_STATE), n_jobs=N_THREADS_PER_MODEL)

    lr_multi_pipeline = Pipeline([('scaler', StandardScaler()), ('logisticregression', LogisticRegression(solver='lbfgs', max_iter=max_iter_lr, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=N_THREADS_PER_MODEL))])

    # Multiclass-specific model configurations
    # GBM doesn't have class_weight - we'll handle via sample_weight in training
    gbm_multi_pipeline = Pipeline([('gradientboostingclassifier', GradientBoostingClassifier(
        random_state=RANDOM_STATE,
        n_estimators=200,  # More estimators for multiclass
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=10,
        subsample=0.8,
        learning_rate=0.05,  # Lower learning rate with more estimators
        validation_fraction=0.1,
        n_iter_no_change=20  # Early stopping
    ))])

    # XGBoost multiclass-specific configuration
    # Note: num_class is set dynamically based on the task data
    # We create a factory function to generate the pipeline with correct num_class
    def create_xgb_multi_pipeline(n_classes):
        """Create XGBoost pipeline with correct number of classes."""
        if XGBClassifier:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('xgbclassifier', XGBClassifier(
                    objective='multi:softprob',
                    num_class=n_classes,
                    random_state=RANDOM_STATE,
                    eval_metric='mlogloss',
                    n_jobs=N_THREADS_PER_MODEL,
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,  # L1 regularization
                    reg_lambda=1.0,  # L2 regularization
                    min_child_weight=3
                ))
            ])
        return None

    # Note: XGBoost and LightGBM pipelines are created dynamically in job building
    # to ensure correct num_class parameter for each task

    # SVM with OvO (One-vs-One) often works better for multiclass
    svm_multi_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            probability=True,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            decision_function_shape='ovo',  # One-vs-One for multiclass
            kernel='rbf',
            C=1.0,
            gamma='scale'
        ))
    ])

    # HistGradientBoostingClassifier - has native class_weight support unlike GBM
    # Generally faster and often better for multiclass
    hist_gbm_multi_pipeline = Pipeline([
        ('histgradientboostingclassifier', HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            class_weight='balanced',  # Native support!
            max_iter=200,
            max_depth=6,
            min_samples_leaf=10,
            learning_rate=0.05,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15
        ))
    ])

    # LightGBM - often outperforms XGBoost for imbalanced multiclass
    def create_lgbm_multi_pipeline(n_classes):
        """Create LightGBM pipeline with correct number of classes."""
        if LGBM_AVAILABLE and LGBMClassifier:
            return Pipeline([
                ('lgbmclassifier', LGBMClassifier(
                    objective='multiclass',
                    num_class=n_classes,
                    random_state=RANDOM_STATE,
                    class_weight='balanced',  # Native support!
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    n_jobs=N_THREADS_PER_MODEL,
                    verbose=-1  # Suppress warnings
                ))
            ])
        return None

    # Random Forest with entropy criterion - sometimes better for multiclass
    rf_entropy_pipeline = Pipeline([
        ('randomforestclassifier', RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight='balanced',  # Use 'balanced' instead of 'balanced_subsample'
            criterion='entropy',  # Information gain often better for multiclass
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=3,
            min_samples_split=5,
            max_features='sqrt',
            n_jobs=N_THREADS_PER_MODEL,
            oob_score=True  # Use OOB for internal validation
        ))
    ])

    # Soft Voting Classifier - combines diverse models for better generalization
    # Created dynamically to include LightGBM when available
    def create_voting_multi_pipeline(n_classes):
        """Create VotingClassifier with diverse base estimators."""
        voting_estimators = [
            ('lr', Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    solver='lbfgs', max_iter=max_iter_lr,
                    random_state=RANDOM_STATE, class_weight='balanced'
                ))
            ])),
            ('rf', RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight='balanced',
                n_estimators=200, max_depth=15, n_jobs=N_THREADS_PER_MODEL
            )),
            ('hist_gbm', HistGradientBoostingClassifier(
                random_state=RANDOM_STATE, class_weight='balanced',
                max_iter=150, max_depth=5, learning_rate=0.05
            )),
        ]
        # Add LightGBM if available
        if LGBM_AVAILABLE and LGBMClassifier:
            voting_estimators.append(('lgbm', LGBMClassifier(
                objective='multiclass', num_class=n_classes,
                random_state=RANDOM_STATE, class_weight='balanced',
                n_estimators=150, max_depth=5, learning_rate=0.05,
                verbose=-1, n_jobs=N_THREADS_PER_MODEL
            )))
        # Add XGBoost if available
        if XGBClassifier:
            voting_estimators.append(('xgb', Pipeline([
                ('scaler', StandardScaler()),
                ('clf', XGBClassifier(
                    objective='multi:softprob', num_class=n_classes,
                    random_state=RANDOM_STATE, n_estimators=150,
                    max_depth=4, learning_rate=0.05, n_jobs=N_THREADS_PER_MODEL
                ))
            ])))

        return VotingClassifier(
            estimators=voting_estimators,
            voting='soft',  # Use probability averaging
            n_jobs=1  # Avoid nested parallelism
        )

    # Binary Voting Classifier - combines diverse models for binary classification
    voting_binary_estimators = [
        ('lr', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                solver='liblinear', max_iter=max_iter_lr,
                random_state=RANDOM_STATE, class_weight='balanced',
                penalty='l1'
            ))
        ])),
        ('rf', RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight='balanced_subsample',
            n_estimators=200, max_depth=15, n_jobs=N_THREADS_PER_MODEL
        )),
        ('hist_gbm', HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, class_weight='balanced',
            max_iter=150, max_depth=5, learning_rate=0.05
        )),
        ('svm', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                probability=True, random_state=RANDOM_STATE,
                class_weight='balanced', kernel='rbf'
            ))
        ])),
    ]
    # Add LightGBM if available (binary)
    if LGBM_AVAILABLE and LGBMClassifier:
        voting_binary_estimators.append(('lgbm', LGBMClassifier(
            objective='binary',
            random_state=RANDOM_STATE, class_weight='balanced',
            n_estimators=150, max_depth=5, learning_rate=0.05,
            verbose=-1, n_jobs=N_THREADS_PER_MODEL
        )))
    # Add XGBoost if available (binary)
    if XGBClassifier:
        voting_binary_estimators.append(('xgb', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(
                objective='binary:logistic',
                random_state=RANDOM_STATE, n_estimators=150,
                max_depth=4, learning_rate=0.05, n_jobs=N_THREADS_PER_MODEL
            ))
        ])))

    voting_binary_pipeline = VotingClassifier(
        estimators=voting_binary_estimators,
        voting='soft',
        n_jobs=1
    )

    # Parameter Grids
    lr_param_grid = {'logisticregression__C': [0.1, 0.5, 1.0, 5.0]}
    rf_param_grid = {'randomforestclassifier__n_estimators': [100, 200], 'randomforestclassifier__max_depth': [10, 20, None], 'randomforestclassifier__min_samples_leaf': [3, 5], 'randomforestclassifier__max_features': ['sqrt', 0.5]}
    gbm_param_grid = {'gradientboostingclassifier__n_estimators': [100, 150], 'gradientboostingclassifier__learning_rate': [0.05, 0.1], 'gradientboostingclassifier__max_depth': [3, 5], 'gradientboostingclassifier__subsample': [0.7, 1.0]}
    svm_param_grid = {'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale', 'auto', 0.1, 1]}
    xgb_param_grid = {}
    if XGBClassifier:
        xgb_param_grid = {'xgbclassifier__n_estimators': [100, 200], 'xgbclassifier__learning_rate': [0.05, 0.1], 'xgbclassifier__max_depth': [3, 5], 'xgbclassifier__subsample': [0.7, 1.0], 'xgbclassifier__colsample_bytree': [0.7, 1.0]}

    # Binary HistGradientBoosting parameter grid
    hist_gbm_binary_param_grid = {
        'histgradientboostingclassifier__max_iter': [150, 250],
        'histgradientboostingclassifier__learning_rate': [0.02, 0.05, 0.1],
        'histgradientboostingclassifier__max_depth': [4, 6, 8],
        'histgradientboostingclassifier__min_samples_leaf': [5, 10, 20],
        'histgradientboostingclassifier__l2_regularization': [0.0, 0.1, 1.0]
    }

    # Binary LightGBM parameter grid
    lgbm_binary_param_grid = {}
    if LGBM_AVAILABLE:
        lgbm_binary_param_grid = {
            'lgbmclassifier__n_estimators': [150, 250],
            'lgbmclassifier__learning_rate': [0.02, 0.05, 0.1],
            'lgbmclassifier__max_depth': [4, 6, 8],
            'lgbmclassifier__num_leaves': [15, 31, 63],
            'lgbmclassifier__min_child_samples': [5, 10, 20],
            'lgbmclassifier__subsample': [0.7, 0.85]
        }

    # Binary Random Forest (Entropy) parameter grid
    rf_entropy_binary_param_grid = {
        'randomforestclassifier__n_estimators': [200, 400],
        'randomforestclassifier__max_depth': [15, 25, None],
        'randomforestclassifier__min_samples_leaf': [2, 5],
        'randomforestclassifier__min_samples_split': [3, 5, 10]
    }

    stacking_param_grid = {
        'lr_stack__logisticregression__C': [0.5, 1.0],
        'rf_stack__randomforestclassifier__max_depth': [10, 20],
        'gbm_stack__gradientboostingclassifier__learning_rate': [0.05, 0.1],
        'svm_stack__svc__C': [0.5, 1.0],
    }
    if XGBClassifier:
        stacking_param_grid['xgb_stack__xgbclassifier__max_depth'] = [3, 5]

    lr_multi_param_grid = {'logisticregression__C': [0.01, 0.1, 1.0, 10.0]}

    # Multiclass-specific parameter grids (more extensive tuning)
    rf_multi_param_grid = {
        'randomforestclassifier__n_estimators': [150, 300, 500],
        'randomforestclassifier__max_depth': [8, 15, 25, None],
        'randomforestclassifier__min_samples_leaf': [2, 5, 10],
        'randomforestclassifier__max_features': ['sqrt', 'log2', 0.3]
    }

    gbm_multi_param_grid = {
        'gradientboostingclassifier__n_estimators': [150, 250],
        'gradientboostingclassifier__learning_rate': [0.02, 0.05, 0.1],
        'gradientboostingclassifier__max_depth': [4, 6, 8],
        'gradientboostingclassifier__min_samples_leaf': [3, 5, 10],
        'gradientboostingclassifier__subsample': [0.7, 0.85]
    }

    svm_multi_param_grid = {
        'svc__C': [0.5, 1.0, 5.0, 10.0],
        'svc__gamma': ['scale', 0.01, 0.1, 1.0]
    }

    xgb_multi_param_grid = {}
    if XGBClassifier:
        xgb_multi_param_grid = {
            'xgbclassifier__n_estimators': [150, 250],
            'xgbclassifier__learning_rate': [0.02, 0.05, 0.1],
            'xgbclassifier__max_depth': [3, 5, 7],
            'xgbclassifier__min_child_weight': [1, 3, 5],
            'xgbclassifier__subsample': [0.7, 0.85],
            'xgbclassifier__colsample_bytree': [0.7, 0.85]
        }

    # HistGradientBoosting parameter grid
    hist_gbm_multi_param_grid = {
        'histgradientboostingclassifier__max_iter': [150, 250],
        'histgradientboostingclassifier__learning_rate': [0.02, 0.05, 0.1],
        'histgradientboostingclassifier__max_depth': [4, 6, 8],
        'histgradientboostingclassifier__min_samples_leaf': [5, 10, 20],
        'histgradientboostingclassifier__l2_regularization': [0.0, 0.1, 1.0]
    }

    # LightGBM parameter grid
    lgbm_multi_param_grid = {}
    if LGBM_AVAILABLE:
        lgbm_multi_param_grid = {
            'lgbmclassifier__n_estimators': [150, 250],
            'lgbmclassifier__learning_rate': [0.02, 0.05, 0.1],
            'lgbmclassifier__max_depth': [4, 6, 8],
            'lgbmclassifier__num_leaves': [15, 31, 63],
            'lgbmclassifier__min_child_samples': [5, 10, 20],
            'lgbmclassifier__subsample': [0.7, 0.85]
        }

    # Random Forest (Entropy) parameter grid
    rf_entropy_param_grid = {
        'randomforestclassifier__n_estimators': [200, 400],
        'randomforestclassifier__max_depth': [15, 25, None],
        'randomforestclassifier__min_samples_leaf': [2, 5],
        'randomforestclassifier__min_samples_split': [3, 5, 10]
    }

    # Voting Classifier - no hyperparameter tuning (uses pre-configured base estimators)
    voting_param_grid = {}

    # Multiclass stacking with better base estimators
    multi_base_estimators_stack = [
        ('lr_stack', clone(lr_multi_pipeline)),
        ('rf_stack', clone(rf_pipeline)),
        ('svm_stack', clone(svm_multi_pipeline))
    ]
    multi_stacking_pipeline = StackingClassifier(
        estimators=multi_base_estimators_stack,
        final_estimator=Pipeline([
            ('scaler_meta', StandardScaler()),
            ('logisticregression_meta', LogisticRegression(
                solver='lbfgs', max_iter=max_iter_lr,
                random_state=RANDOM_STATE, class_weight='balanced'
            ))
        ]),
        passthrough=False,
        cv=StratifiedKFold(n_splits=N_CV_SPLITS_GRIDSEARCH, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=N_THREADS_PER_MODEL
    )

    multi_stacking_param_grid = {
        'lr_stack__logisticregression__C': [0.1, 1.0],
        'rf_stack__randomforestclassifier__max_depth': [10, 20],
        'svm_stack__svc__C': [1.0, 5.0]
    }

    binary_models = [
        ("Logistic Regression (L1)", lr_pipeline, lr_param_grid),
        ("Random Forest (Gini)", rf_pipeline, rf_param_grid),
        ("Random Forest (Entropy)", rf_entropy_binary_pipeline, rf_entropy_binary_param_grid),
        ("Gradient Boosting", gbm_pipeline, gbm_param_grid),
        ("HistGradientBoosting", hist_gbm_binary_pipeline, hist_gbm_binary_param_grid),
        ("SVM (RBF Kernel)", svm_pipeline, svm_param_grid),
    ]
    if xgb_pipeline:
        binary_models.append(("XGBoost", xgb_pipeline, xgb_param_grid))
    if lgbm_binary_pipeline:
        binary_models.append(("LightGBM", lgbm_binary_pipeline, lgbm_binary_param_grid))
    binary_models.append(("Voting Ensemble", voting_binary_pipeline, voting_param_grid))
    binary_models.append(("Stacking Ensemble", stacking_pipeline, stacking_param_grid))

    # MULTICLASS MODELS - Using multiclass-specific configurations
    # Note: Some models (XGBoost, LightGBM, Voting) are added dynamically in job building
    # because they need the correct num_class parameter
    # This list is for reference only - actual models are built dynamically per task
    multi_models_base = [
        ("Logistic Regression (Multinomial)", lr_multi_pipeline, lr_multi_param_grid),
        ("Random Forest (Gini)", rf_pipeline, rf_multi_param_grid),
        ("Random Forest (Entropy)", rf_entropy_pipeline, rf_entropy_param_grid),
        ("Gradient Boosting", gbm_multi_pipeline, gbm_multi_param_grid),
        ("HistGradientBoosting", hist_gbm_multi_pipeline, hist_gbm_multi_param_grid),
        ("SVM (RBF Kernel)", svm_multi_pipeline, svm_multi_param_grid),
        # XGBoost, LightGBM, Voting, Stacking added dynamically in job building
    ]

    # Count expected models (base + XGBoost + LightGBM + Voting + Stacking)
    multi_model_count = len(multi_models_base) + 4  # +4 for XGB, LGBM, Voting, Stacking
    print(f"Defined {len(binary_models)} binary models and ~{multi_model_count} multi-class models (some built dynamically).")

    # --- Build Evaluation Jobs ---
    print("\n=== Building Parallel Evaluation Jobs ===")
    eval_jobs = []

    for label_def, task_specific_data in task_data.items():
        X_train, y_train, X_test, y_test, all_orig_labels, Z_test = task_specific_data
        current_feature_names = feature_names_dict_final.get(label_def)

        if current_feature_names is None or len(current_feature_names) != X_train.shape[1]:
            current_feature_names = [f"Feature_{i+1}" for i in range(X_train.shape[1])]

        is_multi = "Multi:" in label_def

        # Detect class imbalance and provide metric recommendations
        imbalance_info = detect_class_imbalance(y_train)

        # Log class distribution for verification
        if is_multi:
            n_classes = len(all_orig_labels) if all_orig_labels else len(np.unique(y_train))
            train_counts = np.bincount(y_train.astype(int), minlength=n_classes)
            test_counts = np.bincount(y_test.astype(int), minlength=n_classes)
            print(f"  {label_def}:")
            print(f"    Classes: {n_classes}, Train distribution: {dict(enumerate(train_counts))}")
            print(f"    Test distribution: {dict(enumerate(test_counts))}")
        else:
            print(f"  {label_def}:")
            print(f"    Train: {imbalance_info['class_counts']}, Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}:1")

        # Print imbalance warning if detected
        if imbalance_info['is_imbalanced'] and PREFER_PR_AUC_FOR_IMBALANCED:
            print(f"    ⚠️  {imbalance_info['metric_warning']}")

        # Build models based on task type
        if is_multi:
            # Build multiclass models dynamically with correct num_class for XGBoost/LightGBM
            # IMPORTANT: Models that handle class imbalance via sample_weight (GBM, XGBoost)
            # should NOT be wrapped with SMOTE - SMOTE changes data size causing mismatch.
            # Models with native class_weight support CAN use SMOTE for additional balancing.
            task_multi_models = [
                ("Logistic Regression (Multinomial)", create_smote_pipeline(clone(lr_multi_pipeline)), lr_multi_param_grid),
                ("Random Forest (Gini)", create_smote_pipeline(clone(rf_pipeline)), rf_multi_param_grid),
                ("Random Forest (Entropy)", create_smote_pipeline(clone(rf_entropy_pipeline)), rf_entropy_param_grid),
                # GBM uses sample_weight for imbalance - do NOT wrap with SMOTE
                ("Gradient Boosting", clone(gbm_multi_pipeline), gbm_multi_param_grid),
                # HistGBM has native class_weight='balanced' - can use SMOTE
                ("HistGradientBoosting", create_smote_pipeline(clone(hist_gbm_multi_pipeline)), hist_gbm_multi_param_grid),
                ("SVM (RBF Kernel)", create_smote_pipeline(clone(svm_multi_pipeline)), svm_multi_param_grid),
            ]
            # Create XGBoost pipeline - uses sample_weight, do NOT wrap with SMOTE
            if XGBClassifier:
                xgb_task_pipeline = create_xgb_multi_pipeline(n_classes)
                if xgb_task_pipeline:
                    task_multi_models.append(("XGBoost", xgb_task_pipeline, xgb_multi_param_grid))
            # Create LightGBM pipeline - has native class_weight='balanced', can use SMOTE
            if LGBM_AVAILABLE:
                lgbm_task_pipeline = create_lgbm_multi_pipeline(n_classes)
                if lgbm_task_pipeline:
                    task_multi_models.append(("LightGBM", create_smote_pipeline(lgbm_task_pipeline), lgbm_multi_param_grid))
            # VotingClassifier and StackingClassifier are meta-estimators
            # DO NOT wrap with SMOTE - their base estimators already have class_weight='balanced'
            # Wrapping meta-estimators with SMOTE causes issues with their internal fitting logic
            voting_task_pipeline = create_voting_multi_pipeline(n_classes)
            task_multi_models.append(("Voting Ensemble", voting_task_pipeline, voting_param_grid))
            task_multi_models.append(("Stacking Ensemble", clone(multi_stacking_pipeline), multi_stacking_param_grid))
            models = task_multi_models
        else:
            models = binary_models

        for model_name, model_instance, param_grid in models:
            eval_jobs.append((
                label_def, model_name, model_instance, param_grid,
                X_train, y_train, X_test, y_test,
                current_feature_names, is_multi, all_orig_labels
            ))

    print(f"Total evaluation jobs: {len(eval_jobs)}")
    print(f"Running with {N_JOBS_MODELS} parallel workers...")

    # --- RUN PARALLEL EVALUATION ---
    print("\n" + "=" * 60)
    print("STARTING PARALLEL MODEL EVALUATION")
    print("=" * 60 + "\n")

    parallel_start = time.time()

    # Use loky backend - more stable on cluster environments like ARC
    all_results = Parallel(n_jobs=N_JOBS_MODELS, verbose=10, backend='loky')(
        delayed(evaluate_single_model_task)(job) for job in eval_jobs
    )

    parallel_elapsed = time.time() - parallel_start
    print(f"\n=== Parallel Evaluation Complete in {parallel_elapsed:.1f} seconds ===")

    # --- Evaluate Jaakkimainen Benchmark for Binary Dementia vs Normal Tasks ---
    print("\n=== Evaluating Jaakkimainen Benchmark ===")
    for label_def, task_specific_data in task_data.items():
        # Check if this is a binary Dementia vs Normal task (not MCI, not multiclass)
        is_dementia_vs_normal = ("Dementia" in label_def and "Normal" in label_def
                                  and "Multi:" not in label_def and "MCI" not in label_def)

        if is_dementia_vs_normal:
            X_train, y_train, X_test, y_test, all_orig_labels, Z_test = task_specific_data

            if Z_test is not None:
                print(f"  Running Jaakkimainen benchmark for: {label_def}")
                try:
                    benchmark_result = evaluate_benchmark_rule(
                        Z_test=Z_test,
                        y_test=y_test,
                        label_definition=label_def,
                        n_bootstraps=N_BOOTSTRAPS,
                        random_state=RANDOM_STATE
                    )
                    all_results.append(benchmark_result)

                    # Print summary
                    point_auc = benchmark_result["Test_Metrics_Point"].get("auc", np.nan)
                    point_sens = benchmark_result["Test_Metrics_Point"].get("sensitivity", np.nan)
                    point_spec = benchmark_result["Test_Metrics_Point"].get("specificity", np.nan)
                    print(f"    Jaakkimainen AUC: {point_auc:.3f}, Sens: {point_sens:.3f}, Spec: {point_spec:.3f}")
                    print(f"    Benchmark evaluation completed in {benchmark_result['RunTimeSec']:.1f}s")
                except Exception as e:
                    print(f"    ERROR evaluating Jaakkimainen benchmark: {e}")
            else:
                print(f"  Skipping {label_def}: Z_test data not available")

    # --- Process Results ---
    print("\n=== Processing Results ===")

    if not all_results:
        print("No model results generated. Exiting.")
        return

    # Create metrics table
    metrics_df = create_comprehensive_metrics_table(all_results, results_dir)

    # --- Save trained models and pipeline state for reproducibility ---
    print("\n=== Saving Trained Models ===")
    models_dir = os.path.join(base_output_dir, "trained_models")
    os.makedirs(models_dir, exist_ok=True)
    try:
        joblib_dump(all_results, os.path.join(models_dir, "all_results.joblib"))
        joblib_dump(task_data, os.path.join(models_dir, "task_data.joblib"))
        joblib_dump(feature_names_dict_final, os.path.join(models_dir, "feature_names.joblib"))
        print(f"  Saved {len(all_results)} model results to: {models_dir}")
        print(f"  Files: all_results.joblib, task_data.joblib, feature_names.joblib")
        print(f"  These can be loaded with joblib.load() for downstream analysis")
    except Exception as e:
        print(f"  WARNING: Failed to save models: {e}")
        print(f"  Pipeline results are still available in CSV format.")

    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    generate_auc_charts(all_results, charts_dir)
    generate_comparison_charts(all_results, charts_dir)

    # Generate publication-quality visualizations (forest plots, SHAP, ROC, etc.)
    if VISUALIZATIONS_AVAILABLE:
        pub_viz_dir = os.path.join(base_output_dir, "publication_figures")
        os.makedirs(pub_viz_dir, exist_ok=True)
        try:
            run_all_visualizations(
                results_list=all_results,
                task_data=task_data,
                feature_names_dict=feature_names_dict_final,
                output_base_dir=pub_viz_dir
            )
            if SHAP_AVAILABLE:
                print("  SHAP analysis included (install shap: pip install shap)")
            else:
                print("  NOTE: SHAP not installed - SHAP plots skipped (pip install shap)")
        except Exception as e:
            print(f"  WARNING: Publication visualizations failed: {e}")
    else:
        print("  NOTE: Advanced visualizations skipped (publication_visualizations.py not found)")

    # Run statistical comparisons (DeLong, McNemar, feature importance)
    if STATISTICAL_TESTS_AVAILABLE:
        stats_dir = os.path.join(base_output_dir, "statistical_comparisons")
        os.makedirs(stats_dir, exist_ok=True)
        try:
            stat_results = run_statistical_comparisons(
                results_list=all_results,
                task_data=task_data,
                output_dir=stats_dir
            )

            # Extract and save feature importances
            print("\n=== Feature Importance Extraction ===")
            feature_importances = extract_all_feature_importances(all_results, feature_names_dict_final, task_data=task_data)
            for task_name, imp_df in feature_importances.items():
                safe_name = task_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
                imp_path = os.path.join(stats_dir, f"feature_importance_{safe_name}.csv")
                imp_df.to_csv(imp_path, index=False)
                print(f"  Saved feature importance for: {task_name}")

        except Exception as e:
            print(f"  WARNING: Statistical comparisons failed: {e}")
    else:
        print("  NOTE: Statistical comparisons skipped (statistical_tests.py not found)")

    # Run epidemiological analysis (missing data, DCA, calibration metrics, TRIPOD)
    if EPIDEMIOLOGY_UTILS_AVAILABLE:
        epi_dir = os.path.join(base_output_dir, "epidemiology_analysis")
        os.makedirs(epi_dir, exist_ok=True)
        try:
            epi_results = run_epidemiology_analysis(
                data_df=df,  # Use original dataframe to run missing-data diagnostics
                results_list=all_results,
                task_data=task_data,
                feature_names_dict=feature_names_dict_final,
                output_dir=epi_dir
            )
            print("  Epidemiological analysis complete (DCA, calibration metrics, TRIPOD checklist)")
        except Exception as e:
            print(f"  WARNING: Epidemiological analysis failed: {e}")
    else:
        print("  NOTE: Epidemiological analysis skipped (epidemiology_utils.py not found)")

    # Generate thesis-specific tables and figures (Table 1, 1b, 3, 4 and Figures 1-4)
    if THESIS_OUTPUTS_AVAILABLE and run_thesis_outputs is not None:
        try:
            thesis_files = run_thesis_outputs(
                data_df=df,
                results_list=all_results,
                output_base_dir=base_output_dir
            )
            print(f"  Thesis outputs complete: {len(thesis_files)} files generated")
        except Exception as e:
            print(f"  WARNING: Thesis outputs generation failed: {e}")
    else:
        print("  NOTE: Thesis outputs skipped (thesis_tables_figures.py not found)")

    # --- Extract Confusion Matrices for Key Models ---
    print("\n=== Extracting Confusion Matrices ===")
    cm_dir = os.path.join(base_output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    # Define which model/task combinations to extract
    cm_targets = [
        ("XGBoost", "5. Multi: Def Normal vs. Def MCI vs. Def Dementia"),
        ("Random Forest (Entropy)", "1. Def Normal vs. Def Dementia"),
        ("Random Forest (Entropy)", "3. Def Normal vs. Def MCI"),
    ]

    for target_model, target_task in cm_targets:
        # Find matching result
        matching_result = None
        for result in all_results:
            if result.get("ModelName") == target_model and result.get("LabelDefinition") == target_task:
                matching_result = result
                break

        if matching_result is None:
            print(f"  WARNING: Could not find {target_model} for {target_task}")
            continue

        if target_task not in task_data:
            print(f"  WARNING: Task data not found for {target_task}")
            continue

        final_model = matching_result.get("FinalModel")
        if final_model is None:
            print(f"  WARNING: No trained model for {target_model} on {target_task}")
            continue

        try:
            # Get test data
            X_train, y_train, X_test, y_test, all_orig_labels, Z_test = task_data[target_task]

            # Convert labels to int (same as in evaluate_single_model_task)
            y_test_int = y_test.astype(int)
            min_label = np.min(y_test_int)
            if min_label != 0:
                y_test_int = y_test_int - min_label

            # Get predictions
            y_pred = final_model.predict(X_test)

            # Generate confusion matrix
            is_multi = "Multi:" in target_task
            if is_multi:
                labels = [0, 1, 2]
                label_names = ['Normal', 'MCI', 'Dementia']
            else:
                labels = [0, 1]
                if "Dementia" in target_task:
                    label_names = ['Normal', 'Dementia']
                else:
                    label_names = ['Normal', 'MCI']

            cm = confusion_matrix(y_test_int, y_pred, labels=labels)

            # Save as CSV
            safe_name = f"{target_model.replace(' ', '_').replace('(', '').replace(')', '')}_{target_task.replace(' ', '_').replace(':', '').replace('.', '')}"
            cm_df = pd.DataFrame(cm, index=[f"True_{n}" for n in label_names], columns=[f"Pred_{n}" for n in label_names])
            cm_csv_path = os.path.join(cm_dir, f"confusion_matrix_{safe_name}.csv")
            cm_df.to_csv(cm_csv_path)

            # Save as image
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=label_names, yticklabels=label_names,
                   title=f'Confusion Matrix\n{target_model}\n{target_task}',
                   ylabel='True Label',
                   xlabel='Predicted Label')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            cm_img_path = os.path.join(cm_dir, f"confusion_matrix_{safe_name}.png")
            plt.savefig(cm_img_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  Saved: {target_model} on {target_task}")
            print(f"    CSV: {cm_csv_path}")
            print(f"    Image: {cm_img_path}")
            print(f"    Matrix:\n{cm_df.to_string()}\n")

        except Exception as e:
            print(f"  ERROR extracting confusion matrix for {target_model} on {target_task}: {e}")
            import traceback
            traceback.print_exc()

    # --- Final Summary ---
    total_runtime = time.time() - analysis_start_time

    print("\n" + "=" * 60)
    print("PARALLELIZED ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total execution time: {total_runtime / 60:.2f} minutes ({total_runtime:.1f} seconds)")
    print(f"  - Parallel model evaluation: {parallel_elapsed:.1f} seconds")
    print(f"Results saved to: {base_output_dir}")

    # Print summary table
    if not metrics_df.empty:
        print("\n=== RESULTS SUMMARY ===")
        display_cols = ["Task", "Model", "AUC (Point [BS Median (CI)])", "Run Time (s)"]
        valid_cols = [c for c in display_cols if c in metrics_df.columns]
        if valid_cols:
            print(metrics_df[valid_cols].to_string(index=False))


if __name__ == "__main__":
    main()
# fmt: on
